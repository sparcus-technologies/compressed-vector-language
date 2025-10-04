"""
Adaptive Compression Engine - Core compression algorithms with task-aware optimization

This module implements the core compression engine that dynamically adapts compression
strategies based on content analysis, transmission constraints, and task requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CompressionObjective(ABC):
    """Abstract base class for compression objectives"""

    @abstractmethod
    def calculate_loss(
        self, original: np.ndarray, reconstructed: np.ndarray, metadata: Dict[str, Any]
    ) -> float:
        pass


class TaskAwareLoss(CompressionObjective):
    """Task-aware compression loss that prioritizes mission-critical information"""

    def __init__(self, task_weights: Dict[str, float], safety_weight: float = 0.4):
        self.task_weights = task_weights
        self.safety_weight = safety_weight

    def calculate_loss(
        self, original: np.ndarray, reconstructed: np.ndarray, metadata: Dict[str, Any]
    ) -> float:
        """Calculate task-weighted reconstruction loss"""

        # Base reconstruction loss (MSE)
        mse_loss = np.mean((original - reconstructed) ** 2)

        # Task importance weighting
        message_type = metadata.get("message_type", "unknown")
        task_weight = self.task_weights.get(message_type, 1.0)

        # Safety criticality boost
        safety_boost = 1.0
        if message_type in ["emergency", "obstacle"]:
            safety_boost = 1.0 + self.safety_weight
        elif metadata.get("priority") == "critical":
            safety_boost = 1.0 + self.safety_weight * 0.7

        # Semantic preservation penalty
        semantic_penalty = self._calculate_semantic_penalty(
            original, reconstructed, metadata
        )

        total_loss = mse_loss * task_weight * safety_boost + semantic_penalty
        return total_loss

    def _calculate_semantic_penalty(
        self, original: np.ndarray, reconstructed: np.ndarray, metadata: Dict[str, Any]
    ) -> float:
        """Calculate penalty for semantic distortions"""

        # Cosine similarity penalty
        similarity = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed)
        )
        similarity_penalty = max(0, 1 - similarity) * 0.5

        # Magnitude preservation penalty
        orig_magnitude = np.linalg.norm(original)
        recon_magnitude = np.linalg.norm(reconstructed)
        magnitude_penalty = (
            abs(orig_magnitude - recon_magnitude) / max(orig_magnitude, 1e-6) * 0.3
        )

        return similarity_penalty + magnitude_penalty


@dataclass
class CompressionStrategy:
    """Dynamic compression strategy configuration"""

    target_compression_ratio: float
    quality_threshold: float
    bit_budget: int
    semantic_priority_order: List[str]
    error_tolerance: float


class AdaptiveCompressionEngine:
    """Advanced compression engine with dynamic strategy adaptation"""

    def __init__(self, base_config: "CompressionConfig"):
        self.config = base_config
        self.loss_function = TaskAwareLoss(
            {
                "emergency": 2.0,
                "obstacle": 1.8,
                "navigation": 1.4,
                "coordination": 1.2,
                "status": 1.0,
            }
        )

        # Learned compression strategies
        self.compression_strategies = {
            "emergency": CompressionStrategy(
                target_compression_ratio=5.0,  # Less aggressive for safety
                quality_threshold=0.9,
                bit_budget=128,
                semantic_priority_order=[
                    "action",
                    "spatial",
                    "temporal",
                    "status",
                    "uncertainty",
                ],
                error_tolerance=0.05,
            ),
            "navigation": CompressionStrategy(
                target_compression_ratio=8.0,
                quality_threshold=0.8,
                bit_budget=96,
                semantic_priority_order=[
                    "spatial",
                    "action",
                    "temporal",
                    "status",
                    "uncertainty",
                ],
                error_tolerance=0.1,
            ),
            "status": CompressionStrategy(
                target_compression_ratio=15.0,  # More aggressive for status
                quality_threshold=0.7,
                bit_budget=64,
                semantic_priority_order=[
                    "status",
                    "temporal",
                    "spatial",
                    "action",
                    "uncertainty",
                ],
                error_tolerance=0.2,
            ),
            "default": CompressionStrategy(
                target_compression_ratio=10.0,
                quality_threshold=0.75,
                bit_budget=80,
                semantic_priority_order=[
                    "action",
                    "spatial",
                    "temporal",
                    "status",
                    "uncertainty",
                ],
                error_tolerance=0.15,
            ),
        }

        # Neural compression components
        self.semantic_projectors = {}
        self.reconstruction_networks = {}

    def initialize_neural_components(self, embedding_dim: int):
        """Initialize learnable neural compression components"""

        for semantic_type in ["action", "spatial", "temporal", "status", "uncertainty"]:
            # Semantic-specific dimension reduction
            self.semantic_projectors[semantic_type] = nn.Sequential(
                nn.Linear(embedding_dim // 5, self.config.base_compressed_dim // 5),
                nn.ReLU(),
                nn.Linear(
                    self.config.base_compressed_dim // 5,
                    self.config.base_compressed_dim // 5,
                ),
                nn.Tanh(),
            )

            # Reconstruction networks
            self.reconstruction_networks[semantic_type] = nn.Sequential(
                nn.Linear(self.config.base_compressed_dim // 5, embedding_dim // 5),
                nn.ReLU(),
                nn.Linear(embedding_dim // 5, embedding_dim // 5),
            )

    def adaptive_compress(
        self, semantic_vectors: Dict[str, np.ndarray], message_metadata: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Adaptively compress semantic vectors based on content analysis"""

        message_type = message_metadata.get("message_type", "default")
        strategy = self.compression_strategies.get(
            message_type, self.compression_strategies["default"]
        )

        compressed_vectors = {}
        total_bit_usage = 0

        # Prioritize semantic types based on message content
        priority_order = strategy.semantic_priority_order
        remaining_budget = strategy.bit_budget

        for semantic_type in priority_order:
            if semantic_type not in semantic_vectors or remaining_budget <= 0:
                continue

            vector = semantic_vectors[semantic_type]

            # Allocate bits based on priority and remaining budget
            if semantic_type == priority_order[0]:  # Highest priority
                allocated_bits = min(remaining_budget // 2, len(vector) * 8)
            else:
                allocated_bits = min(
                    remaining_budget
                    // (len(priority_order) - priority_order.index(semantic_type)),
                    len(vector) * 4,
                )

            # Compress based on bit allocation
            compressed_vector = self._compress_vector_adaptive(
                vector, allocated_bits, strategy.quality_threshold
            )

            compressed_vectors[semantic_type] = compressed_vector
            remaining_budget -= allocated_bits
            total_bit_usage += allocated_bits

        # Add metadata about compression strategy used
        compressed_vectors["_compression_metadata"] = np.array(
            [
                strategy.target_compression_ratio,
                strategy.quality_threshold,
                total_bit_usage,
                len(priority_order),
            ],
            dtype=np.float32,
        )

        return compressed_vectors

    def _compress_vector_adaptive(
        self, vector: np.ndarray, bit_budget: int, quality_threshold: float
    ) -> np.ndarray:
        """Compress individual vector with adaptive quality control"""

        # Calculate target dimensions based on bit budget
        target_dims = max(1, bit_budget // 16)  # Assuming float16
        target_dims = min(target_dims, len(vector))

        if target_dims >= len(vector):
            return vector.astype(np.float16)

        # Use truncated SVD for dimensionality reduction
        try:
            U, s, Vt = np.linalg.svd(vector.reshape(1, -1), full_matrices=False)

            # Select components to meet quality threshold
            cumulative_variance = np.cumsum(s**2) / np.sum(s**2)
            components_needed = (
                np.searchsorted(cumulative_variance, quality_threshold) + 1
            )
            components_needed = min(components_needed, target_dims)

            # Reconstruct with selected components
            compressed = (U[:, :components_needed] * s[:components_needed]) @ Vt[
                :components_needed, :
            ]
            return compressed.flatten().astype(np.float16)

        except np.linalg.LinAlgError:
            # Fallback to simple truncation
            return vector[:target_dims].astype(np.float16)

    def adaptive_decompress(
        self, compressed_vectors: Dict[str, np.ndarray], original_dims: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """Adaptively decompress vectors with error correction"""

        decompressed_vectors = {}

        # Extract compression metadata
        compression_metadata = compressed_vectors.get(
            "_compression_metadata", np.array([10.0, 0.75, 64, 5])
        )

        for semantic_type, compressed_vector in compressed_vectors.items():
            if semantic_type == "_compression_metadata":
                continue

            original_dim = original_dims.get(semantic_type, len(compressed_vector))

            # Decompress with error correction
            decompressed = self._decompress_vector_adaptive(
                compressed_vector, original_dim, compression_metadata
            )

            decompressed_vectors[semantic_type] = decompressed

        return decompressed_vectors

    def _decompress_vector_adaptive(
        self, compressed_vector: np.ndarray, target_dim: int, metadata: np.ndarray
    ) -> np.ndarray:
        """Decompress individual vector with adaptive reconstruction"""

        if len(compressed_vector) >= target_dim:
            return compressed_vector[:target_dim]

        # Intelligent padding/interpolation for dimension expansion
        expansion_ratio = target_dim / len(compressed_vector)

        if expansion_ratio <= 2.0:
            # Linear interpolation for small expansions
            indices = np.linspace(0, len(compressed_vector) - 1, target_dim)
            decompressed = np.interp(
                indices, np.arange(len(compressed_vector)), compressed_vector
            )
        else:
            # Repeat + noise injection for larger expansions
            repeats = int(np.ceil(expansion_ratio))
            expanded = np.tile(compressed_vector, repeats)[:target_dim]

            # Add small amount of structured noise to avoid exact repetitions
            noise_scale = np.std(compressed_vector) * 0.1
            noise = np.random.normal(0, noise_scale, target_dim)
            decompressed = expanded + noise

        return decompressed.astype(np.float32)

    def optimize_compression_strategy(
        self,
        messages: List[Dict[str, Any]],
        semantic_vectors_batch: List[Dict[str, np.ndarray]],
    ):
        """Learn optimal compression strategies from data"""

        print("Optimizing compression strategies...")

        # Group by message type
        type_groups = {}
        for i, msg in enumerate(messages):
            msg_type = msg["message_type"]
            if msg_type not in type_groups:
                type_groups[msg_type] = []
            type_groups[msg_type].append((msg, semantic_vectors_batch[i]))

        # Optimize strategy for each message type
        for msg_type, data in type_groups.items():
            if len(data) < 10:  # Need sufficient samples
                continue

            best_strategy = self._optimize_single_strategy(data)
            self.compression_strategies[msg_type] = best_strategy

        print(f"Optimized strategies for {len(type_groups)} message types")

    def _optimize_single_strategy(
        self, data: List[Tuple[Dict, Dict[str, np.ndarray]]]
    ) -> CompressionStrategy:
        """Optimize compression strategy for specific message type"""

        # Test different compression parameters
        best_score = float("inf")
        best_strategy = self.compression_strategies["default"]

        # Parameter grid search
        compression_ratios = [5.0, 8.0, 12.0, 15.0, 20.0]
        quality_thresholds = [0.6, 0.7, 0.8, 0.9]
        bit_budgets = [48, 64, 96, 128]

        for ratio in compression_ratios:
            for threshold in quality_thresholds:
                for budget in bit_budgets:
                    # Create candidate strategy
                    candidate = CompressionStrategy(
                        target_compression_ratio=ratio,
                        quality_threshold=threshold,
                        bit_budget=budget,
                        semantic_priority_order=[
                            "action",
                            "spatial",
                            "temporal",
                            "status",
                            "uncertainty",
                        ],
                        error_tolerance=0.1,
                    )

                    # Evaluate on sample data
                    score = self._evaluate_strategy(
                        candidate, data[: min(20, len(data))]
                    )

                    if score < best_score:
                        best_score = score
                        best_strategy = candidate

        return best_strategy

    def _evaluate_strategy(
        self,
        strategy: CompressionStrategy,
        data: List[Tuple[Dict, Dict[str, np.ndarray]]],
    ) -> float:
        """Evaluate compression strategy on data samples"""

        total_loss = 0.0
        total_compression_ratio = 0.0

        for msg, semantic_vectors in data:
            try:
                # Apply compression strategy
                compressed = self.adaptive_compress(semantic_vectors, msg)

                # Calculate original and compressed sizes
                original_size = sum(
                    len(v) * 4 for v in semantic_vectors.values()
                )  # float32
                compressed_size = sum(
                    len(v) * 2
                    for k, v in compressed.items()
                    if k != "_compression_metadata"
                )  # float16

                compression_ratio = original_size / max(compressed_size, 1)

                # Calculate reconstruction quality
                original_dims = {k: len(v) for k, v in semantic_vectors.items()}
                decompressed = self.adaptive_decompress(compressed, original_dims)

                # Calculate task-aware loss
                for semantic_type, original_vector in semantic_vectors.items():
                    if semantic_type in decompressed:
                        reconstructed = decompressed[semantic_type][
                            : len(original_vector)
                        ]
                        loss = self.loss_function.calculate_loss(
                            original_vector, reconstructed, msg
                        )
                        total_loss += loss

                total_compression_ratio += compression_ratio

            except Exception:
                # Penalize strategies that cause errors
                total_loss += 10.0

        # Combined score: balance compression ratio and quality
        avg_loss = total_loss / len(data)
        avg_compression = total_compression_ratio / len(data)

        # Penalize if compression ratio doesn't meet target
        compression_penalty = (
            max(0, strategy.target_compression_ratio - avg_compression) * 0.5
        )

        return avg_loss + compression_penalty
