"""Configuration management for Q-KVComm"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class QKVCommConfig:
    """Configuration for Q-KVComm system"""

    # Quantization settings
    target_bits: float = 6.0
    min_bits: int = 4
    max_bits: int = 8
    quantization_enabled: bool = True
    profiling_samples: int = 100

    # Calibration settings
    calibration_enabled: bool = True
    calibration_samples: int = 50
    use_scalar_calibration: bool = False  # Auto-set for heterogeneous models

    # KV selection settings
    layer_selection_ratio: float = 0.7
    attention_weight: float = 0.5  # alpha in paper
    gaussian_mu_ratio: float = 0.5  # μ as ratio of total layers
    gaussian_sigma_ratio: float = 0.15  # σ as ratio of total layers

    # ⭐ NEW: Information extraction settings
    extraction_method: Literal["yake", "spacy_ner", "hybrid", "simple"] = "yake"
    extraction_max_tokens: int = 50
    extraction_min_confidence: float = 0.5
    extraction_cache_enabled: bool = True

    # ⭐ NEW: Memory management settings
    max_memory_mb: float = 1024.0
    enable_disk_cache: bool = True
    adaptive_compression: bool = True

    # Cache settings
    cache_dir: Optional[str] = None
    use_cache: bool = True

    # Mode selection
    mode: Literal["baseline", "quantization_only", "hybrid", "full"] = "full"

    # Hybrid mode settings (DEPRECATED - now using extraction_method)
    hybrid_entity_extraction: Optional[str] = None  # Kept for backward compatibility
    hybrid_max_entity_tokens: int = 20

    def __post_init__(self):
        """Validate configuration"""
        assert (
            self.min_bits <= self.target_bits <= self.max_bits
        ), f"Invalid bit configuration: {self.min_bits} <= {self.target_bits} <= {self.max_bits}"
        assert (
            0 < self.layer_selection_ratio <= 1.0
        ), f"Layer selection ratio must be in (0, 1], got {self.layer_selection_ratio}"
        assert (
            0 <= self.attention_weight <= 1.0
        ), f"Attention weight must be in [0, 1], got {self.attention_weight}"

        # Backward compatibility: map old hybrid_entity_extraction to extraction_method
        if self.hybrid_entity_extraction is not None:
            method_map = {
                "yake": "yake",
                "qa": "simple",  # QA requires query, fallback to simple
                "attention": "simple",
                "attention_queryless": "simple",
                "simple": "simple",
            }
            self.extraction_method = method_map.get(
                self.hybrid_entity_extraction, "yake"
            )

        # Update mode-based settings
        if self.mode == "baseline":
            self.quantization_enabled = False
            self.calibration_enabled = False
        elif self.mode == "quantization_only":
            self.calibration_enabled = False

    @property
    def compression_ratio(self) -> float:
        """Expected compression ratio relative to FP16"""
        if not self.quantization_enabled:
            return 1.0
        return 16.0 / self.target_bits
