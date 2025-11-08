"""
Experiment 1: Compression vs Quality Trade-off

Tests different compression levels (target bits) and measures quality preservation.
Shows the relationship between compression ratio and answer quality metrics.

Usage:
    python exp1_compression_quality.py --max-samples 100
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

from q_kvcomm import QKVCommConfig, QKVCommSystem


class CompressionQualityExperiment(ExperimentBase):
    """Experiment 1: Compression vs Quality Trade-off"""

    def run(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Run compression quality trade-off experiment"""
        self.print_header("EXPERIMENT 1: COMPRESSION VS QUALITY TRADE-OFF")

        target_bits_list = [4.0, 6.0, 8.0]
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

        for target_bits in target_bits_list:
            self.print_section(f"Testing with {target_bits}-bit quantization")

            config = QKVCommConfig(
                mode="hybrid",
                quantization_enabled=True,
                calibration_enabled=True,
                target_bits=target_bits,
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
                output_dir=str(self.output_dir / f"exp1_bits_{target_bits}"),
                enable_baseline=True,
            )

            benchmark.run_benchmark(
                dataset_names=["squad", "hotpot_qa", "narrativeqa"],
                max_samples=min(50, self.max_samples),
            )

            # Get results from benchmark.results
            dataset_results = benchmark.results

            # Aggregate results
            for dataset_name, dataset_metrics in dataset_results.items():
                successful = [r for r in dataset_metrics if r.get("success", False)]
                if successful:
                    results.append(
                        {
                            "target_bits": target_bits,
                            "dataset": dataset_name,
                            "num_samples": len(successful),
                            "contextual_relevance": np.mean(
                                [r["contextual_relevance"] for r in successful]
                            ),
                            "answer_completeness": np.mean(
                                [r["answer_completeness"] for r in successful]
                            ),
                            "semantic_fidelity": np.mean(
                                [r["semantic_fidelity"] for r in successful]
                            ),
                            "response_coherence": np.mean(
                                [r["response_coherence"] for r in successful]
                            ),
                            "compression_ratio": np.mean(
                                [r["compression_ratio"] for r in successful]
                            ),
                            "bandwidth_saved_mb": np.sum(
                                [r["bits_saved"] for r in successful]
                            )
                            / 1e6,  # Convert bits to MB
                            "avg_inference_time": np.mean(
                                [r["inference_time"] for r in successful]
                            ),
                            "compression_quality": np.mean(
                                [
                                    r.get("compression_quality_score", 1.0)
                                    for r in successful
                                ]
                            ),
                            "semantic_preservation": np.mean(
                                [
                                    r.get("semantic_preservation", 1.0)
                                    for r in successful
                                ]
                            ),
                        }
                    )

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "experiment_1_compression_quality.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Summary
        print("\n📊 Summary:")
        print(
            df.groupby("target_bits")
            .agg(
                {
                    "contextual_relevance": "mean",
                    "compression_ratio": "mean",
                    "bandwidth_saved_mb": "sum",
                }
            )
            .round(4)
        )

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Compression vs Quality Trade-off"
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
    experiment = CompressionQualityExperiment(
        output_dir=args.output_dir,
        device=args.device,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )

    experiment.run(model_name=args.model)


if __name__ == "__main__":
    main()
