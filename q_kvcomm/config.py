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
    use_scalar_calibration: bool = (
        False  # Use dimension-agnostic calibration for heterogeneous models
    )

    # KV selection settings
    layer_selection_ratio: float = 0.7
    attention_weight: float = 0.5  # alpha in paper
    gaussian_mu_ratio: float = 0.5  # μ as ratio of total layers
    gaussian_sigma_ratio: float = 0.15  # σ as ratio of total layers

    # Cache settings
    cache_dir: Optional[str] = None
    use_cache: bool = True

    # Mode selection
    mode: Literal["baseline", "quantization_only", "full"] = "full"

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
