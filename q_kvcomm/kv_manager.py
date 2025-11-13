"""KV cache extraction and layer selection"""

import torch
import numpy as np
from typing import List, Tuple

class KVCacheManager:
    """Manages KV cache extraction and layer selection"""
    
    def __init__(self, config):
        self.config = config
        
    def compute_attention_importance(
        self,
        model,
        tokenizer,
        context: str,
        device: str
    ) -> np.ndarray:
        """Compute attention importance scores for each layer (Eq 2 from paper)"""
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, use_cache=False)
        
        attentions = outputs.attentions
        
        if attentions is None or len(attentions) == 0:
            num_layers = model.config.num_hidden_layers
            return np.ones(num_layers) / num_layers
        
        importance_scores = []
        for layer_attention in attentions:
            if layer_attention is None:
                importance_scores.append(0.0)
            else:
                # Eq 2: S_l^a = (1/HT) * sum of attention weights
                avg_attention = layer_attention.mean().item()
                importance_scores.append(avg_attention)
        
        scores = np.array(importance_scores)
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = np.ones_like(scores) / len(scores)
        
        return scores
    
    def compute_gaussian_prior(self, num_layers: int) -> np.ndarray:
        """
        Compute Gaussian prior over layers (Eq 3 from paper)
        
        P_l = exp(-(l - μ)² / (2σ²))
        """
        mu = self.config.gaussian_mu_ratio * num_layers
        sigma = self.config.gaussian_sigma_ratio * num_layers
        
        layers = np.arange(num_layers)
        prior = np.exp(-((layers - mu) ** 2) / (2 * sigma ** 2))
        
        return prior / prior.sum()
    
    def select_layers(
        self,
        model,
        tokenizer,
        context: str,
        device: str
    ) -> List[int]:
        """
        Select layers using attention-Gaussian strategy (Eq 3 from paper)
        
        S_l = α * S_l^a + (1-α) * P_l
        """
        num_layers = model.config.num_hidden_layers
        
        # Compute attention importance (Eq 2)
        attention_scores = self.compute_attention_importance(model, tokenizer, context, device)
        
        # Compute Gaussian prior
        gaussian_prior = self.compute_gaussian_prior(num_layers)
        
        # Combine (Eq 3)
        alpha = self.config.attention_weight
        combined_scores = alpha * attention_scores + (1 - alpha) * gaussian_prior
        
        # Select top layers
        num_selected = max(1, int(self.config.layer_selection_ratio * num_layers))
        selected_indices = np.argsort(combined_scores)[::-1][:num_selected]
        
        return sorted(selected_indices.tolist())
    
    def extract_kv_cache(
        self,
        model,
        tokenizer,
        context: str,
        device: str
    ) -> Tuple[tuple, List[int]]:
        """
        Extract KV cache with layer selection
        
        Returns:
            (past_key_values, selected_layer_indices)
        """
        model.eval()
        
        # First, select layers
        selected_layers = self.select_layers(model, tokenizer, context, device)
        
        # Then extract KV cache
        with torch.no_grad():
            inputs = tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Return only selected layers' KV caches
        selected_kv = tuple(past_key_values[i] for i in selected_layers)
        
        return selected_kv, selected_layers
    
    def compute_cache_size(self, kv_cache: tuple) -> int:
        """Compute total size of KV cache in bytes"""
        total_size = 0
        for key, value in kv_cache:
            total_size += key.numel() * key.element_size()
            total_size += value.numel() * value.element_size()
        return total_size