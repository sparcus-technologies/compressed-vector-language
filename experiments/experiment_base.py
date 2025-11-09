"""
Base class for Q-KVComm experiments

Provides common functionality for all experiments.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ExperimentBase:
    """Base class for all Q-KVComm experiments"""

    def __init__(
        self,
        output_dir: str = "experiment_results",
        device: str = "auto",
        max_samples_per_dataset: int = 100,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect best available device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.max_samples = max_samples_per_dataset
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"💻 Device: {self.device}")
        if self.device == "mps":
            print("   Using Apple Silicon GPU acceleration")
        elif self.device == "cpu":
            print("   Running on CPU (slower but works everywhere)")

    def print_header(self, title: str):
        """Print experiment header"""
        print("=" * 80)
        print(title)
        print("=" * 80)

    def print_section(self, title: str):
        """Print section separator"""
        print(f"\n{'─' * 80}")
        print(title)
        print(f"{'─' * 80}")