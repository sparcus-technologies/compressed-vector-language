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

        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_samples = max_samples_per_dataset
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

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
