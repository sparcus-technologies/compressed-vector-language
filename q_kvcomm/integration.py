"""Main Q-KVComm system integration with proper extraction and REAL compression"""

import io
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .adaptive_extraction import ContextTypeDetector, InformationExtractor
from .calibration import CalibrationModule
from .config import QKVCommConfig
from .kv_manager import KVCacheManager
from .memory_manager import AdaptiveCompressionManager, MemoryManager
from .quantization import QuantizationEngine


class QKVCommSystem:
    """Complete Q-KVComm system with REAL compression and transmission"""

    def __init__(
        self,
        sender_model,
        receiver_model,
        config: Optional[QKVCommConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize Q-KVComm system

        Args:
            sender_model: Sender model with tokenizer
            receiver_model: Receiver model with tokenizer
            config: Configuration
            device: Device to use
        """
        self.sender = sender_model
        self.receiver = receiver_model
        self.config = config or QKVCommConfig()
        self.device = device

        # Detect heterogeneous model architectures and enable scalar calibration if needed
        self._detect_and_configure_calibration()

        # Initialize components
        self.quantizer = QuantizationEngine(self.config)
        self.calibrator = CalibrationModule(self.config)
        self.kv_manager = KVCacheManager(self.config)

        # Information extraction
        self.extractor = InformationExtractor(
            extraction_method=self.config.extraction_method
        )
        self.context_detector = ContextTypeDetector()

        # Memory management
        self.memory_manager = MemoryManager(
            max_memory_mb=self.config.max_memory_mb,
            cache_dir=Path(self.config.cache_dir) if self.config.cache_dir else None,
            enable_disk_cache=self.config.enable_disk_cache,
        )

        if self.config.adaptive_compression:
            self.compression_manager = AdaptiveCompressionManager(self.memory_manager)
        else:
            self.compression_manager = None

        self.selected_layers: Optional[List[int]] = None
        self._is_calibrated = False

        # Detect if model is chat-tuned
        self.is_chat_model = self._detect_chat_model()

    def _detect_and_configure_calibration(self):
        """
        Detect if sender and receiver have different head dimensions.
        If so, enable scalar calibration mode for compatibility.
        """
        if not self.config.calibration_enabled:
            return

        # Get head dimensions from both models
        sender_head_dim = (
            self.sender.config.hidden_size // self.sender.config.num_attention_heads
        )
        receiver_head_dim = (
            self.receiver.config.hidden_size // self.receiver.config.num_attention_heads
        )

        # If head dimensions differ, we must use scalar calibration
        if sender_head_dim != receiver_head_dim:
            self.config.use_scalar_calibration = True

    def _detect_chat_model(self) -> bool:
        """Detect if model is instruction/chat tuned"""
        model_name = getattr(self.sender, "name_or_path", "").lower()
        chat_keywords = ["chat", "instruct", "zephyr", "assistant"]
        return any(kw in model_name for kw in chat_keywords)

    def _format_prompt(
        self, context: str, question: str, extracted_facts: str = ""
    ) -> str:
        """Format prompt based on model type with extracted facts"""
        if self.is_chat_model:
            # Chat format for instruction-tuned models
            if extracted_facts:
                return f"""<|system|>
You are a helpful assistant that answers questions based on the given context.</s>
<|user|>
Key Information: {extracted_facts}

Context: {context}

Question: {question}</s>
<|assistant|>
"""
            else:
                return f"""<|system|>
You are a helpful assistant that answers questions based on the given context.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
        else:
            # Simple format for base models
            if extracted_facts:
                return f"""Key Facts: {extracted_facts}

Context: {context}

Question: {question}
Answer:"""
            else:
                return f"""Context: {context}

Question: {question}
Answer:"""

    def calibrate(self, calibration_data: List[str]):
        """
        Perform one-time calibration

        Args:
            calibration_data: List of calibration texts
        """
        print("=" * 60)
        print("CALIBRATION PHASE")
        print("=" * 60)

        # Use a sample context to determine layer selection
        sample_context = calibration_data[0]
        _, self.selected_layers = self.kv_manager.extract_kv_cache(
            self.sender, self.sender.tokenizer, sample_context, self.device
        )

        print(
            f"Selected {len(self.selected_layers)} / "
            f"{self.sender.config.num_hidden_layers} layers"
        )
        print(f"Selected layers: {self.selected_layers}")

        # Quantization profiling
        if self.config.quantization_enabled:
            print("\n" + "-" * 60)
            print("QUANTIZATION PROFILING")
            print("-" * 60)

            sensitivity = self.quantizer.compute_sensitivity(
                self.sender, calibration_data, self.selected_layers, self.device
            )

            self.quantizer.allocate_bits(sensitivity)

            print(f"\nBit allocation:")
            for layer in sorted(self.quantizer.bit_allocation.keys())[:5]:
                bits = self.quantizer.bit_allocation[layer]
                sens = sensitivity[layer]
                print(f"  Layer {layer:2d}: {bits}-bit (sensitivity: {sens:.6f})")
            print(f"  ... (showing first 5 layers)")

        # Feature calibration
        if self.config.calibration_enabled:
            print("\n" + "-" * 60)
            print("FEATURE CALIBRATION")
            print("-" * 60)

            self.calibrator.calibrate_sender_receiver(
                self.sender,
                self.receiver,
                calibration_data,
                self.selected_layers,
                self.device,
            )

        self._is_calibrated = True
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60 + "\n")

    def _pack_quantized_tensor(self, tensor: torch.Tensor, num_bits: int) -> bytes:
        """
        ⭐ FIXED: Chunk-based packing to avoid NumPy memory limits
        
        Args:
            tensor: Quantized integer tensor (int32)
            num_bits: Bit width (4, 6, or 8)
            
        Returns:
            Packed bytes
        """
        # Work directly on CPU tensor to avoid data transfer overhead
        if tensor.is_cuda or tensor.device.type == 'mps':
            tensor = tensor.cpu()
        
        shape = tensor.shape
        total_elements = tensor.numel()
        
        # Process in chunks to avoid memory issues (1M elements per chunk)
        chunk_size = 1_000_000
        
        if num_bits == 8:
            # 8-bit: direct conversion chunk-by-chunk
            result_bytes = bytearray()
            tensor_flat = tensor.flatten()
            
            for i in range(0, total_elements, chunk_size):
                chunk = tensor_flat[i:i+chunk_size]
                result_bytes.extend(chunk.numpy().astype(np.uint8).tobytes())
            
            return bytes(result_bytes)
        
        elif num_bits == 4:
            # 4-bit: pack two values per byte
            num_packed = (total_elements + 1) // 2
            result_bytes = bytearray(num_packed)
            tensor_flat = tensor.flatten()
            
            for i in range(0, total_elements, chunk_size):
                chunk = tensor_flat[i:i+chunk_size].numpy().astype(np.int32)
                chunk_start_idx = i // 2
                
                for j in range(0, len(chunk), 2):
                    low = chunk[j] & 0x0F
                    high = (chunk[j + 1] & 0x0F) if j + 1 < len(chunk) else 0
                    result_bytes[chunk_start_idx + j // 2] = low | (high << 4)
            
            return bytes(result_bytes)
        
        elif num_bits == 6:
            # 6-bit: pack 4 values into 3 bytes
            num_groups = (total_elements + 3) // 4
            num_packed_bytes = num_groups * 3
            result_bytes = bytearray(num_packed_bytes)
            tensor_flat = tensor.flatten()
            
            for i in range(0, total_elements, chunk_size):
                chunk = tensor_flat[i:i+chunk_size].numpy().astype(np.int32)
                chunk_group_start = i // 4
                
                for j in range(0, len(chunk), 4):
                    v0 = chunk[j] & 0x3F if j < len(chunk) else 0
                    v1 = chunk[j + 1] & 0x3F if j + 1 < len(chunk) else 0
                    v2 = chunk[j + 2] & 0x3F if j + 2 < len(chunk) else 0
                    v3 = chunk[j + 3] & 0x3F if j + 3 < len(chunk) else 0
                    
                    group_idx = chunk_group_start + j // 4
                    result_bytes[group_idx * 3] = v0 | ((v1 & 0x03) << 6)
                    result_bytes[group_idx * 3 + 1] = (v1 >> 2) | ((v2 & 0x0F) << 4)
                    result_bytes[group_idx * 3 + 2] = (v2 >> 4) | (v3 << 2)
            
            return bytes(result_bytes)
        
        else:
            # Fallback: use 8-bit
            return tensor.flatten().numpy().astype(np.uint8).tobytes()

    def _unpack_quantized_tensor(
        self, data_bytes: bytes, shape: tuple, num_bits: int, device: str
    ) -> torch.Tensor:
        """
        ⭐ FIXED: Chunk-based unpacking
        
        Args:
            data_bytes: Packed bytes
            shape: Original tensor shape
            num_bits: Bit width
            device: Target device
            
        Returns:
            Unpacked int32 tensor
        """
        total_elements = int(np.prod(shape))
        chunk_size = 1_000_000
        
        if num_bits == 8:
            # 8-bit: direct conversion in chunks
            result = torch.empty(total_elements, dtype=torch.int32)
            
            for i in range(0, total_elements, chunk_size):
                end_idx = min(i + chunk_size, total_elements)
                chunk_data = np.frombuffer(
                    data_bytes[i:end_idx], 
                    dtype=np.uint8
                ).astype(np.int32)
                result[i:end_idx] = torch.from_numpy(chunk_data)
        
        elif num_bits == 4:
            # 4-bit: unpack pairs
            result = torch.empty(total_elements, dtype=torch.int32)
            
            packed = np.frombuffer(data_bytes, dtype=np.uint8)
            for i in range(0, total_elements, chunk_size):
                end_idx = min(i + chunk_size, total_elements)
                packed_start = i // 2
                packed_end = (end_idx + 1) // 2
                
                chunk_packed = packed[packed_start:packed_end]
                for j, byte_val in enumerate(chunk_packed):
                    idx = i + j * 2
                    if idx < total_elements:
                        result[idx] = byte_val & 0x0F
                    if idx + 1 < total_elements:
                        result[idx + 1] = (byte_val >> 4) & 0x0F
        
        elif num_bits == 6:
            # 6-bit: unpack groups of 4 from 3 bytes
            result = torch.empty(total_elements, dtype=torch.int32)
            packed = np.frombuffer(data_bytes, dtype=np.uint8)
            
            for i in range(0, total_elements, chunk_size):
                end_idx = min(i + chunk_size, total_elements)
                
                for idx in range(i, end_idx, 4):
                    group_idx = idx // 4
                    if group_idx * 3 + 2 < len(packed):
                        b0 = packed[group_idx * 3]
                        b1 = packed[group_idx * 3 + 1]
                        b2 = packed[group_idx * 3 + 2]
                        
                        if idx < total_elements:
                            result[idx] = b0 & 0x3F
                        if idx + 1 < total_elements:
                            result[idx + 1] = ((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F
                        if idx + 2 < total_elements:
                            result[idx + 2] = ((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F
                        if idx + 3 < total_elements:
                            result[idx + 3] = (b2 >> 2) & 0x3F
        
        else:
            # Fallback
            result = torch.from_numpy(
                np.frombuffer(data_bytes[:total_elements], dtype=np.uint8).astype(np.int32)
            )
        
        # Reshape and move to device
        result = result[:total_elements].reshape(shape)
        return result.to(device)

    def _serialize_quantized_data(self, quantized_data: dict, metadata: dict) -> bytes:
        """
        Efficient serialization using bit-packing
        
        Args:
            quantized_data: Quantized key/value tensors
            metadata: Quantization metadata
            
        Returns:
            Serialized bytes
        """
        if not metadata.get('quantized', False):
            # Not quantized - fallback to pickle (shouldn't happen in quantization mode)
            import pickle
            return pickle.dumps({'key': quantized_data['key'], 'value': quantized_data['value'], 'metadata': metadata})
        
        # Get bit width
        num_bits = metadata['num_bits']
        
        # Pack key and value separately
        key_packed = self._pack_quantized_tensor(quantized_data['key'], num_bits)
        value_packed = self._pack_quantized_tensor(quantized_data['value'], num_bits)
        
        # Create compact header
        key_shape = quantized_data['key'].shape
        value_shape = quantized_data['value'].shape
        
        header = struct.pack(
            'B4I4Iffff',  # Byte, 4 ints, 4 ints, 4 floats
            num_bits,
            *key_shape,
            *value_shape,
            metadata['key_scale'],
            metadata['key_zero_point'],
            metadata['value_scale'],
            metadata['value_zero_point']
        )
        
        # Combine: header + key_data + value_data
        return header + key_packed + value_packed
    
    def _deserialize_quantized_data(self, data_bytes: bytes, device: str) -> Tuple[dict, dict]:
        """
        Efficient deserialization
        
        Args:
            data_bytes: Serialized bytes
            device: Target device
            
        Returns:
            (quantized_data, metadata)
        """
        # Parse header
        header_size = struct.calcsize('B4I4Iffff')
        header_data = struct.unpack('B4I4Iffff', data_bytes[:header_size])
        
        num_bits = header_data[0]
        key_shape = header_data[1:5]
        value_shape = header_data[5:9]
        key_scale = header_data[9]
        key_zp = header_data[10]
        val_scale = header_data[11]
        val_zp = header_data[12]
        
        # Calculate sizes
        key_elements = int(np.prod(key_shape))
        value_elements = int(np.prod(value_shape))
        
        if num_bits == 8:
            key_packed_size = key_elements
            value_packed_size = value_elements
        elif num_bits == 4:
            key_packed_size = (key_elements + 1) // 2
            value_packed_size = (value_elements + 1) // 2
        elif num_bits == 6:
            key_packed_size = ((key_elements + 3) // 4) * 3
            value_packed_size = ((value_elements + 3) // 4) * 3
        else:
            key_packed_size = key_elements
            value_packed_size = value_elements
        
        # Extract packed data
        key_start = header_size
        key_end = key_start + key_packed_size
        value_end = key_end + value_packed_size
        
        key_packed = data_bytes[key_start:key_end]
        value_packed = data_bytes[key_end:value_end]
        
        # Unpack
        key_tensor = self._unpack_quantized_tensor(key_packed, key_shape, num_bits, device)
        value_tensor = self._unpack_quantized_tensor(value_packed, value_shape, num_bits, device)
        
        quantized_data = {
            'key': key_tensor,
            'value': value_tensor
        }
        
        metadata = {
            'quantized': True,
            'num_bits': num_bits,
            'key_scale': key_scale,
            'key_zero_point': key_zp,
            'value_scale': val_scale,
            'value_zero_point': val_zp,
            'shape': key_shape,
            'dtype': 'int32'
        }
        
        return quantized_data, metadata

    def communicate(
        self, context: str, query: str, max_new_tokens: int = 50
    ) -> Tuple[str, Dict]:
        """
        Execute Q-KVComm communication with REAL compression and transmission

        Args:
            context: Context from sender
            query: Query for receiver
            max_new_tokens: Max tokens to generate

        Returns:
            Tuple of (generated_text, metrics)
        """
        if not self._is_calibrated:
            raise RuntimeError("System not calibrated. Call calibrate() first.")

        metrics = {
            "compression_ratios": [],
            "total_bits_original": 0,
            "total_bits_compressed": 0,
            "total_bytes_transmitted": 0,
            "num_layers_transmitted": 0,
        }

        # STEP 0: Extract important information
        cached_facts = None
        if self.config.extraction_cache_enabled:
            cached_facts = self.memory_manager.get(context, prefix="facts")

        if cached_facts is not None:
            extracted_facts_list = cached_facts
            metrics["extraction_cache_hit"] = True
        else:
            # Detect context type for adaptive extraction
            context_type = self.context_detector.detect(context)
            metrics["context_type"] = context_type

            # Extract facts
            extracted_facts_list = self.extractor.extract_facts(context)

            # Cache the extraction
            if self.config.extraction_cache_enabled:
                self.memory_manager.set(context, extracted_facts_list, prefix="facts")

            metrics["extraction_cache_hit"] = False

        # Format facts for transmission
        extracted_facts_text = self.extractor.format_facts_for_transmission(
            extracted_facts_list,
            max_tokens=self.config.extraction_max_tokens,
            min_confidence=self.config.extraction_min_confidence,
        )

        metrics["num_facts_extracted"] = len(extracted_facts_list)
        metrics["facts_text_length"] = len(extracted_facts_text)
        metrics["extracted_facts"] = extracted_facts_text

        # Step 1: Extract KV cache from sender
        self.sender.eval()
        with torch.no_grad():
            sender_inputs = self.sender.tokenizer(
                context, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            sender_outputs = self.sender(**sender_inputs, use_cache=True)
            sender_past_kv = sender_outputs.past_key_values

        selected_layers = self.selected_layers

        # Step 2: Process, QUANTIZE, SERIALIZE, DESERIALIZE, DEQUANTIZE (simulating transmission)
        sender_kv_for_receiver = {}
        
        print("\n" + "=" * 60)
        print("TRANSMISSION SIMULATION")
        print("=" * 60)

        for layer_idx in selected_layers:
            if layer_idx >= len(sender_past_kv):
                continue

            key, value = sender_past_kv[layer_idx]

            # Apply calibration
            if self.config.calibration_enabled:
                key, value = self.calibrator.apply_calibration(key, value, layer_idx)

            # Quantize
            temp_past_kv = list(sender_past_kv)
            temp_past_kv[layer_idx] = (key, value)
            temp_past_kv = tuple(temp_past_kv)

            quantized_data, metadata = self.quantizer.quantize_kv_cache(
                temp_past_kv, layer_idx
            )

            # ⭐ SERIALIZE with efficient bit-packing
            transmitted_bytes = self._serialize_quantized_data(quantized_data, metadata)
            transmitted_size = len(transmitted_bytes)
            metrics["total_bytes_transmitted"] += transmitted_size

            # ⭐ FIXED: Calculate original size WITHOUT creating arrays that overflow
            try:
                # Method 1: Direct calculation from tensor properties
                original_size_bytes = (
                    key.numel() * key.element_size() + 
                    value.numel() * value.element_size()
                )
            except (RuntimeError, OverflowError) as e:
                # Method 2: Fallback to shape-based estimation
                try:
                    key_size = int(np.prod(key.shape)) * 4  # Assume float32
                    value_size = int(np.prod(value.shape)) * 4
                    original_size_bytes = key_size + value_size
                except:
                    # Method 3: Last resort - estimate from transmitted size
                    original_size_bytes = transmitted_size * 4  # Conservative estimate
            
            # ⭐ DESERIALIZE
            received_quantized_data, received_metadata = self._deserialize_quantized_data(
                transmitted_bytes, self.device
            )

            # Metrics
            actual_compression_ratio = original_size_bytes / max(transmitted_size, 1)
            metrics["compression_ratios"].append(actual_compression_ratio)

            metrics["total_bits_original"] += original_size_bytes * 8
            metrics["total_bits_compressed"] += transmitted_size * 8

            print(f"Layer {layer_idx:2d}: {original_size_bytes:,} bytes → {transmitted_size:,} bytes "
                  f"({actual_compression_ratio:.2f}x compression)")

            # Dequantize
            key_deq, value_deq = self.quantizer.dequantize_kv_cache(
                received_quantized_data, received_metadata
            )

            sender_kv_for_receiver[layer_idx] = (key_deq, value_deq)

        metrics["num_layers_transmitted"] = len(sender_kv_for_receiver)
        
        print(f"\nTotal transmitted: {metrics['total_bytes_transmitted']:,} bytes "
              f"({metrics['total_bytes_transmitted']/1024/1024:.2f} MB)")
        print("=" * 60 + "\n")

        # Step 3: Format prompt with extracted facts
        full_prompt = self._format_prompt(context, query, extracted_facts_text)

        # Step 4: Receiver processes query WITH sender context integrated
        self.receiver.eval()

        # Use hooks to inject sender KV cache
        hooks = []

        def create_hook(layer_idx, sender_key, sender_value):
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states = output[0]
                    present = output[1]

                    if isinstance(present, tuple) and len(present) == 2:
                        recv_key, recv_value = present

                        batch_size = recv_key.shape[0]
                        num_heads = recv_key.shape[1]
                        head_dim = recv_key.shape[3]

                        # Expand sender KV to match batch size
                        sk = sender_key
                        sv = sender_value
                        if sk.shape[0] != batch_size:
                            sk = sk.expand(batch_size, -1, -1, -1)
                            sv = sv.expand(batch_size, -1, -1, -1)

                        # Verify dimensions match
                        if sk.shape[1] == num_heads and sk.shape[3] == head_dim:
                            # Concatenate along sequence dimension
                            integrated_key = torch.cat([sk, recv_key], dim=2)
                            integrated_value = torch.cat([sv, recv_value], dim=2)

                            new_present = (integrated_key, integrated_value)
                            return (hidden_states, new_present) + output[2:]

                return output

            return hook_fn

        # Register hooks for selected layers
        layer_modules = []
        if hasattr(self.receiver, "transformer"):
            # GPT-2 style
            layer_modules = self.receiver.transformer.h
        elif hasattr(self.receiver, "model"):
            if hasattr(self.receiver.model, "layers"):
                # Llama/TinyLlama style
                layer_modules = self.receiver.model.layers
            elif hasattr(self.receiver.model, "decoder"):
                layer_modules = self.receiver.model.decoder.layers

        for layer_idx, (sender_key, sender_value) in sender_kv_for_receiver.items():
            if layer_idx < len(layer_modules):
                hook = layer_modules[layer_idx].register_forward_hook(
                    create_hook(layer_idx, sender_key, sender_value)
                )
                hooks.append(hook)

        # Generate with integrated KV
        with torch.no_grad():
            receiver_inputs = self.receiver.tokenizer(
                full_prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            attention_mask = torch.ones_like(receiver_inputs["input_ids"])

            gen_outputs = self.receiver.generate(
                input_ids=receiver_inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.receiver.tokenizer.eos_token_id,
                eos_token_id=self.receiver.tokenizer.eos_token_id,
            )

            generated_text = self.receiver.tokenizer.decode(
                gen_outputs[0][len(receiver_inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Compute final metrics
        if metrics["compression_ratios"]:
            metrics["avg_compression_ratio"] = sum(metrics["compression_ratios"]) / len(
                metrics["compression_ratios"]
            )
        else:
            metrics["avg_compression_ratio"] = 1.0

        metrics["overall_compression_ratio"] = metrics["total_bits_original"] / max(
            metrics["total_bits_compressed"], 1
        )
        
        # Bandwidth saved calculation
        original_mb = metrics["total_bits_original"] / 8 / 1024 / 1024
        transmitted_mb = metrics["total_bytes_transmitted"] / 1024 / 1024
        metrics["bandwidth_saved_mb"] = original_mb - transmitted_mb

        # Memory stats
        metrics["memory_stats"] = self.memory_manager.get_stats()

        return generated_text.strip(), metrics