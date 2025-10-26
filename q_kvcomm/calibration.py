"""Feature space calibration for heterogeneous models"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

class CalibrationModule:
    """Handles zero-shot feature space calibration"""
    
    def __init__(self, config):
        self.config = config
        self.sender_stats: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
        self.receiver_stats: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
        
    def compute_statistics(
        self,
        model,
        calibration_data: list,
        selected_layers: list,
        device: str = "cuda"
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Compute first and second-order statistics (Eq 10-13)
        
        Args:
            model: Model to compute statistics for
            calibration_data: List of calibration texts
            selected_layers: Layers to calibrate
            device: Device to run on
            
        Returns:
            Dictionary mapping layer idx to {'key_mean', 'key_std', 'value_mean', 'value_std'}
        """
        model.eval()
        
        # Accumulators for online statistics
        stats = {
            layer: {
                'key_sum': None,
                'key_sq_sum': None,
                'value_sum': None,
                'value_sq_sum': None,
                'count': 0
            } for layer in selected_layers
        }
        
        print(f"Computing calibration statistics with {len(calibration_data)} samples...")
        
        with torch.no_grad():
            for text in tqdm(calibration_data[:self.config.calibration_samples]):
                inputs = model.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512).to(device)
                
                outputs = model(**inputs, output_hidden_states=True, use_cache=True)
                past_key_values = outputs.past_key_values
                
                for layer_idx in selected_layers:
                    if layer_idx >= len(past_key_values):
                        continue
                    
                    key, value = past_key_values[layer_idx]
                    
                    # Online mean and variance computation
                    if stats[layer_idx]['key_sum'] is None:
                        stats[layer_idx]['key_sum'] = torch.zeros_like(key[0, 0, 0, :])
                        stats[layer_idx]['key_sq_sum'] = torch.zeros_like(key[0, 0, 0, :])
                        stats[layer_idx]['value_sum'] = torch.zeros_like(value[0, 0, 0, :])
                        stats[layer_idx]['value_sq_sum'] = torch.zeros_like(value[0, 0, 0, :])
                    
                    # Aggregate over batch, heads, and sequence dimensions
                    key_flat = key.mean(dim=(0, 1, 2))  # Average over batch, heads, seq
                    value_flat = value.mean(dim=(0, 1, 2))
                    
                    stats[layer_idx]['key_sum'] += key_flat
                    stats[layer_idx]['key_sq_sum'] += key_flat ** 2
                    stats[layer_idx]['value_sum'] += value_flat
                    stats[layer_idx]['value_sq_sum'] += value_flat ** 2
                    stats[layer_idx]['count'] += 1
        
        # Compute final statistics
        final_stats = {}
        for layer_idx in selected_layers:
            if stats[layer_idx]['count'] == 0:
                continue
                
            n = stats[layer_idx]['count']
            
            # Mean
            key_mean = stats[layer_idx]['key_sum'] / n
            value_mean = stats[layer_idx]['value_sum'] / n
            
            # Standard deviation
            key_var = (stats[layer_idx]['key_sq_sum'] / n) - (key_mean ** 2)
            value_var = (stats[layer_idx]['value_sq_sum'] / n) - (value_mean ** 2)
            
            # Add small epsilon for numerical stability
            key_std = torch.sqrt(torch.clamp(key_var, min=1e-8))
            value_std = torch.sqrt(torch.clamp(value_var, min=1e-8))
            
            final_stats[layer_idx] = {
                'key_mean': key_mean.cpu(),
                'key_std': key_std.cpu(),
                'value_mean': value_mean.cpu(),
                'value_std': value_std.cpu()
            }
        
        return final_stats
    
    def calibrate_sender_receiver(
        self,
        sender_model,
        receiver_model,
        calibration_data: list,
        selected_layers: list,
        device: str = "cuda"
    ):
        """
        Compute calibration statistics for both models
        
        Args:
            sender_model: Sender model
            receiver_model: Receiver model
            calibration_data: Calibration texts
            selected_layers: Layers to calibrate
            device: Device
        """
        print("Computing sender statistics...")
        self.sender_stats = self.compute_statistics(
            sender_model, calibration_data, selected_layers, device
        )
        
        print("Computing receiver statistics...")
        self.receiver_stats = self.compute_statistics(
            receiver_model, calibration_data, selected_layers, device
        )
    
    def apply_calibration(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply calibration transform (Eq 14)
        
        Transform: KV_calibrated = (KV_s - μ_s) / σ_s * σ_r + μ_r
        
        Args:
            key: Sender key tensor
            value: Sender value tensor
            layer_idx: Layer index
            
        Returns:
            Calibrated (key, value)
        """
        if not self.config.calibration_enabled:
            return key, value
            
        if self.sender_stats is None or self.receiver_stats is None:
            return key, value
        
        if layer_idx not in self.sender_stats or layer_idx not in self.receiver_stats:
            return key, value
        
        sender_s = self.sender_stats[layer_idx]
        receiver_s = self.receiver_stats[layer_idx]
        
        device = key.device
        
        # Get statistics on correct device
        key_mean_s = sender_s['key_mean'].to(device)
        key_std_s = sender_s['key_std'].to(device)
        key_mean_r = receiver_s['key_mean'].to(device)
        key_std_r = receiver_s['key_std'].to(device)
        
        value_mean_s = sender_s['value_mean'].to(device)
        value_std_s = sender_s['value_std'].to(device)
        value_mean_r = receiver_s['value_mean'].to(device)
        value_std_r = receiver_s['value_std'].to(device)
        
        # Apply affine transformation
        # Broadcast over batch, heads, sequence dimensions
        key_calibrated = (key - key_mean_s) / key_std_s * key_std_r + key_mean_r
        value_calibrated = (value - value_mean_s) / value_std_s * value_std_r + value_mean_r
        
        return key_calibrated, value_calibrated