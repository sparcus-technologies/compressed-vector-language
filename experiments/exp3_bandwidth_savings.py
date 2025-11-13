"""
Experiment 3: Bandwidth Savings Analysis

Quantifies bandwidth reduction across different layer selection ratios.
Shows the trade-off between communication cost and quality preservation.

Usage:
    python exp3_bandwidth_savings.py --max-samples 100
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_suite import BenchmarkSuite
from experiment_base import ExperimentBase

from q_kvcomm import QKVCommConfig
from q_kvcomm.integration import QKVCommSystem


class BandwidthSavingsExperiment(ExperimentBase):
    """Experiment 3: Bandwidth Savings Analysis"""

    def run(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Run bandwidth savings analysis experiment"""
        self.print_header("EXPERIMENT 3: BANDWIDTH SAVINGS ANALYSIS")

        results = []

        # Load model
        print(f"\nLoading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.tokenizer = tokenizer
        print("✓ Model loaded\n")

        # Test different layer selection ratios
        layer_selection_ratios = [0.5, 0.7, 0.9, 1.0]

        for ratio in layer_selection_ratios:
            self.print_section(f"Testing with {ratio*100:.0f}% layer selection")

            config = QKVCommConfig(
                mode="hybrid",
                quantization_enabled=True,
                calibration_enabled=True,
                target_bits=6.0,
                layer_selection_ratio=ratio,
                extraction_method="yake",
                max_memory_mb=1024,
            )

            # Create Q-KVComm system
            qkvcomm_system = QKVCommSystem(
                sender_model=model,
                receiver_model=model,
                config=config,
                device=self.device,
            )

            # Run benchmark
            benchmark = BenchmarkSuite(
                qkvcomm_system=qkvcomm_system,
                output_dir=str(self.output_dir / f"exp3_layers_{int(ratio*100)}"),
                enable_baseline=True,
            )

            dataset_results = benchmark.run_benchmark(
                dataset_names=["squad", "narrativeqa"],
                max_samples=min(30, self.max_samples),
            )

            # Aggregate results
            for dataset_name, dataset_metrics in dataset_results.items():
                successful = [r for r in dataset_metrics if r.get("success", False)]
                if successful:
                    total_bandwidth_saved = (
                        np.sum([r["bits_saved"] for r in successful]) / 1e6
                    )  # Convert bits to MB
                    avg_compression = np.mean(
                        [r["compression_ratio"] for r in successful]
                    )

                    results.append(
                        {
                            "layer_selection_ratio": ratio,
                            "dataset": dataset_name,
                            "num_samples": len(successful),
                            "compression_ratio": avg_compression,
                            "total_bandwidth_saved_mb": total_bandwidth_saved,
                            "avg_bandwidth_per_sample_mb": total_bandwidth_saved
                            / len(successful),
                            "contextual_relevance": np.mean(
                                [r["contextual_relevance"] for r in successful]
                            ),
                            "answer_completeness": np.mean(
                                [r["answer_completeness"] for r in successful]
                            ),
                            "avg_layers_transmitted": np.mean(
                                [r.get("num_layers_transmitted", 0) for r in successful]
                            ),
                        }
                    )

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "experiment_3_bandwidth_savings.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Summary
        print("\n📊 Summary:")
        print(
            df.groupby("layer_selection_ratio")
            .agg(
                {
                    "compression_ratio": "mean",
                    "total_bandwidth_saved_mb": "sum",
                    "contextual_relevance": "mean",
                }
            )
            .round(4)
        )

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Bandwidth Savings Analysis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run experiment on",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100, help="Maximum samples per dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model to use for experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Run experiment
    experiment = BandwidthSavingsExperiment(
        output_dir=args.output_dir,
        device=args.device,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )

    experiment.run(model_name=args.model)


if __name__ == "__main__":
    main()
