"""Main Q-KVComm system integration"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .calibration import CalibrationModule
from .config import QKVCommConfig
from .entity_extraction import EntityExtractor
from .kv_manager import KVCacheManager
from .quantization import QuantizationEngine


class QKVCommSystem:
    """Complete Q-KVComm system"""

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

        # Initialize components
        self.quantizer = QuantizationEngine(self.config)
        self.calibrator = CalibrationModule(self.config)
        self.kv_manager = KVCacheManager(self.config)

        # Initialize entity extractor for hybrid mode
        if self.config.mode == "hybrid":
            self.entity_extractor = EntityExtractor(
                model=self.sender, tokenizer=self.sender.tokenizer
            )
        else:
            self.entity_extractor = None

        self.selected_layers: Optional[List[int]] = None
        self._is_calibrated = False

        # Detect if model is chat-tuned
        self.is_chat_model = self._detect_chat_model()

    def _detect_chat_model(self) -> bool:
        """Detect if model is instruction/chat tuned"""
        model_name = getattr(self.sender, "name_or_path", "").lower()
        chat_keywords = ["chat", "instruct", "zephyr", "assistant"]
        return any(kw in model_name for kw in chat_keywords)

    def _format_prompt(self, context: str, question: str) -> str:
        """Format prompt based on model type"""
        if self.is_chat_model:
            # Chat format for instruction-tuned models
            return f"""<|system|>
You are a helpful assistant that answers questions based on the given context.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
        else:
            # Simple format for base models
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
        print(f"Mode: {self.config.mode}")
        print(
            f"Quantization: {'Enabled' if self.config.quantization_enabled else 'Disabled'}"
        )
        print(
            f"Calibration: {'Enabled' if self.config.calibration_enabled else 'Disabled'}"
        )

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

    def communicate(
        self, context: str, query: str, max_new_tokens: int = 50
    ) -> Tuple[str, Dict]:
        """
        Execute Q-KVComm communication

        Args:
            context: Context from sender
            query: Query for receiver
            max_new_tokens: Max tokens to generate

        Returns:
            Tuple of (generated_text, metrics)
        """
        if not self._is_calibrated:
            raise RuntimeError("System not calibrated. Call calibrate() first.")

        # TRUE BASELINE MODE: Give receiver full context as text (no KV cache)
        if self.config.mode == "baseline":
            return self._communicate_baseline(context, query, max_new_tokens)

        metrics = {
            "compression_ratios": [],
            "total_bits_original": 0,
            "total_bits_compressed": 0,
            "num_layers_transmitted": 0,
        }

        # Step 1: Extract KV cache from sender (using context only)
        self.sender.eval()
        with torch.no_grad():
            sender_inputs = self.sender.tokenizer(
                context, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            sender_outputs = self.sender(**sender_inputs, use_cache=True)
            sender_past_kv = sender_outputs.past_key_values

        selected_layers = self.selected_layers

        # Step 2: Process selected layers (calibrate + quantize + dequantize)
        sender_kv_for_receiver = {}

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

            # Metrics
            stats = self.quantizer.get_compression_stats(metadata)
            metrics["compression_ratios"].append(stats["compression_ratio"])

            original_size = key.numel() * 2 + value.numel() * 2
            if metadata.get("quantized", False):
                compressed_size = (
                    (key.numel() + value.numel()) * metadata["num_bits"] / 8
                )
            else:
                compressed_size = original_size

            metrics["total_bits_original"] += original_size * 8
            metrics["total_bits_compressed"] += compressed_size * 8

            # Dequantize
            key_deq, value_deq = self.quantizer.dequantize_kv_cache(
                quantized_data, metadata
            )

            sender_kv_for_receiver[layer_idx] = (key_deq, value_deq)

        metrics["num_layers_transmitted"] = len(sender_kv_for_receiver)

        # Step 3: Format prompt - HYBRID MODE or pure KV
        if self.config.mode == "hybrid":
            # HYBRID: Extract key facts from context and include them as text
            if self.config.hybrid_entity_extraction == "qa":
                extracted_facts = self.entity_extractor.extract_with_qa_model(
                    context=context,
                    query=query,
                    max_tokens=self.config.hybrid_max_entity_tokens,
                    device=self.device,
                    qa_model_name=self.config.hybrid_qa_model,
                )
            else:
                extracted_facts = self.entity_extractor.extract(
                    context=context,
                    query=query,
                    max_tokens=self.config.hybrid_max_entity_tokens,
                    method=self.config.hybrid_entity_extraction,
                    device=self.device,
                )

            # Receiver gets extracted facts + query (KV provides deeper context)
            if self.is_chat_model:
                receiver_prompt = f"""<|system|>
You are a helpful assistant that answers questions based on context.</s>
<|user|>
Context: {extracted_facts}

Question: {query}</s>
<|assistant|>
"""
            else:
                receiver_prompt = (
                    f"Context: {extracted_facts}\nQuestion: {query}\nAnswer:"
                )

            # Track text size for metrics
            text_tokens = len(self.receiver.tokenizer.encode(extracted_facts))
            metrics["hybrid_text_tokens"] = text_tokens
            metrics["hybrid_extracted_facts"] = extracted_facts
        else:
            # PURE KV: Receiver gets ONLY the question (context comes from KV cache)
            if self.is_chat_model:
                receiver_prompt = f"""<|system|>
You are a helpful assistant that answers questions based on context.</s>
<|user|>
{query}</s>
<|assistant|>
"""
            else:
                receiver_prompt = f"Question: {query}\nAnswer:"

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
                receiver_prompt, return_tensors="pt", truncation=True, max_length=512
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

        return generated_text.strip(), metrics

    def _communicate_baseline(
        self, context: str, query: str, max_new_tokens: int = 50
    ) -> Tuple[str, Dict]:
        """
        Baseline communication: Receiver gets full context as TEXT (no KV cache tricks)
        This is what normal LLM inference looks like.

        Args:
            context: Context text
            query: Query text
            max_new_tokens: Max tokens to generate

        Returns:
            Tuple of (generated_text, metrics)
        """
        # Format prompt with full context as text
        if self.is_chat_model:
            full_prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the given context.</s>
<|user|>
Context: {context}

Question: {query}</s>
<|assistant|>
"""
        else:
            full_prompt = f"""Context: {context}

Question: {query}
Answer:"""

        # Normal generation (no KV cache injection)
        self.receiver.eval()
        with torch.no_grad():
            inputs = self.receiver.tokenizer(
                full_prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            outputs = self.receiver.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.receiver.tokenizer.eos_token_id,
                eos_token_id=self.receiver.tokenizer.eos_token_id,
            )

            generated_text = self.receiver.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )

        # Baseline metrics (no compression)
        metrics = {
            "compression_ratios": [1.0],
            "total_bits_original": 0,
            "total_bits_compressed": 0,
            "num_layers_transmitted": 0,
            "avg_compression_ratio": 1.0,
            "overall_compression_ratio": 1.0,
        }

        return generated_text.strip(), metrics
