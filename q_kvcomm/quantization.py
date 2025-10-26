"""Adaptive layer-aware quantization for KV caches"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

class QuantizationEngine:
    """Handles adaptive quantization of KV caches"""
    
    def __init__(self, config):
        self.config = config
        self.sensitivity_map: Optional[Dict[int, float]] = None
        self.bit_allocation: Optional[Dict[int, int]] = None
        
    def quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        num_bits: int
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Asymmetric per-tensor quantization
        
        Args:
            tensor: Input tensor to quantize
            num_bits: Number of bits for quantization
            
        Returns:
            quantized_tensor: Quantized values
            scale: Scale factor
            zero_point: Zero point
        """
        if tensor.numel() == 0:
            return tensor, 1.0, 0.0
            
        # Handle constant tensors
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        
        if t_max - t_min < 1e-8:
            scale = 1.0
            zero_point = 0.0
            quantized = torch.zeros_like(tensor, dtype=torch.int32)
        else:
            # Compute scale and zero point (Eq 5-6)
            qmax = 2 ** num_bits - 1
            scale = (t_max - t_min) / qmax
            zero_point = -t_min / scale
            
            # Quantize (Eq 7)
            quantized = torch.clamp(
                torch.round(tensor / scale + zero_point),
                0, qmax
            ).to(torch.int32)
        
        return quantized, scale, zero_point
    
    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: float,
        zero_point: float
    ) -> torch.Tensor:
        """
        Dequantize tensor (Eq 8)
        
        Args:
            quantized: Quantized tensor
            scale: Scale factor
            zero_point: Zero point
            
        Returns:
            Dequantized tensor
        """
        return (quantized.float() - zero_point) * scale
    
    def compute_sensitivity(
        self,
        model,
        calibration_data: list,
        selected_layers: list,
        device: str = "cuda"
    ) -> Dict[int, float]:
        """
        Profile quantization sensitivity for each layer (Algorithm 1)
        
        Args:
            model: The model to profile
            calibration_data: List of input texts
            selected_layers: Layers to profile
            device: Device to run on
            
        Returns:
            Dictionary mapping layer indices to sensitivity scores
        """
        model.eval()
        sensitivity = {layer: 0.0 for layer in selected_layers}
        test_bits = self.config.min_bits  # Use minimum bits for testing
        
        print(f"Profiling quantization sensitivity with {len(calibration_data)} samples...")
        
        with torch.no_grad():
            for text in tqdm(calibration_data[:self.config.profiling_samples]):
                # Get KV caches from model
                inputs = model.tokenizer(text, return_tensors="pt", 
                                       truncation=True, max_length=512).to(device)
                
                outputs = model(**inputs, output_hidden_states=True, 
                              use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Measure reconstruction error for each layer
                for layer_idx in selected_layers:
                    if layer_idx >= len(past_key_values):
                        continue
                        
                    key, value = past_key_values[layer_idx]
                    
                    # Quantize and dequantize
                    key_q, key_scale, key_zp = self.quantize_tensor(key, test_bits)
                    key_deq = self.dequantize_tensor(key_q, key_scale, key_zp)
                    
                    value_q, val_scale, val_zp = self.quantize_tensor(value, test_bits)
                    value_deq = self.dequantize_tensor(value_q, val_scale, val_zp)
                    
                    # Compute MSE (Eq 18)
                    key_error = torch.mean((key - key_deq) ** 2).item()
                    value_error = torch.mean((value - value_deq) ** 2).item()
                    
                    sensitivity[layer_idx] += (key_error + value_error)
        
        # Normalize by number of samples
        num_samples = min(len(calibration_data), self.config.profiling_samples)
        for layer in sensitivity:
            sensitivity[layer] /= num_samples
            
        self.sensitivity_map = sensitivity
        return sensitivity
    
    def allocate_bits(self, sensitivity_map: Dict[int, float]) -> Dict[int, int]:
        """
        Allocate bit-widths based on sensitivity rankings
        
        Strategy from paper:
        - Top 30% most sensitive: max_bits (8-bit)
        - Bottom 30% least sensitive: min_bits (4-bit)
        - Middle 40%: target_bits (6-bit)
        
        Args:
            sensitivity_map: Layer sensitivity scores
            
        Returns:
            Dictionary mapping layer indices to bit-widths
        """
        layers = sorted(sensitivity_map.keys())
        sensitivities = [sensitivity_map[l] for l in layers]
        
        # Sort layers by sensitivity
        sorted_indices = np.argsort(sensitivities)[::-1]  # Descending order
        
        num_layers = len(layers)
        top_30_count = int(0.3 * num_layers)
        bottom_30_count = int(0.3 * num_layers)
        
        bit_allocation = {}
        
        for i, idx in enumerate(sorted_indices):
            layer = layers[idx]
            if i < top_30_count:
                # Top 30% - most sensitive
                bit_allocation[layer] = self.config.max_bits
            elif i >= num_layers - bottom_30_count:
                # Bottom 30% - least sensitive
                bit_allocation[layer] = self.config.min_bits
            else:
                # Middle 40%
                bit_allocation[layer] = int(self.config.target_bits)
        
        self.bit_allocation = bit_allocation
        
        # Compute actual average
        avg_bits = np.mean(list(bit_allocation.values()))
        print(f"Allocated bits - Target: {self.config.target_bits:.1f}, "
              f"Actual: {avg_bits:.1f}")
        
        return bit_allocation
    
    def quantize_kv_cache(
        self,
        past_key_values: tuple,
        layer_idx: int
    ) -> Tuple[dict, dict]:
        """
        Quantize KV cache for a specific layer
        
        Args:
            past_key_values: KV cache tuple from model
            layer_idx: Layer index to quantize
            
        Returns:
            Tuple of (quantized_data, metadata)
        """
        if not self.config.quantization_enabled or self.bit_allocation is None:
            # Return original in expected format
            key, value = past_key_values[layer_idx]
            return {
                'key': key,
                'value': value
            }, {
                'quantized': False
            }
        
        num_bits = self.bit_allocation.get(layer_idx, int(self.config.target_bits))
        key, value = past_key_values[layer_idx]
        
        # Quantize key and value
        key_q, key_scale, key_zp = self.quantize_tensor(key, num_bits)
        value_q, val_scale, val_zp = self.quantize_tensor(value, num_bits)
        
        quantized_data = {
            'key': key_q,
            'value': value_q
        }
        
        metadata = {
            'quantized': True,
            'num_bits': num_bits,
            'key_scale': key_scale,
            'key_zero_point': key_zp,
            'value_scale': val_scale,
            'value_zero_point': val_zp,
            'shape': key.shape,
            'dtype': str(key.dtype)
        }
        
        return quantized_data, metadata
    
    def dequantize_kv_cache(
        self,
        quantized_data: dict,
        metadata: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize KV cache
        
        Args:
            quantized_data: Quantized key and value
            metadata: Quantization metadata
            
        Returns:
            Tuple of (key, value)
        """
        if not metadata.get('quantized', False):
            return quantized_data['key'], quantized_data['value']
        
        key_deq = self.dequantize_tensor(
            quantized_data['key'],
            metadata['key_scale'],
            metadata['key_zero_point']
        )
        
        value_deq = self.dequantize_tensor(
            quantized_data['value'],
            metadata['value_scale'],
            metadata['value_zero_point']
        )
        
        return key_deq, value_deq
    
    def get_compression_stats(self, metadata: dict) -> dict:
        """Calculate compression statistics"""
        if not metadata.get('quantized', False):
            return {'compression_ratio': 1.0, 'original_bits': 16, 'compressed_bits': 16}
        
        original_bits = 16  # FP16/BF16
        compressed_bits = metadata['num_bits']
        compression_ratio = original_bits / compressed_bits
        
        return {
            'compression_ratio': compression_ratio,
            'original_bits': original_bits,
            'compressed_bits': compressed_bits
        }