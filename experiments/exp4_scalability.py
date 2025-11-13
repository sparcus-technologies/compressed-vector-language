"""
Experiment 4: Scalability Study

Tests Q-KVComm across different model sizes to evaluate scalability.
Shows how the system performs with models of varying parameter counts.

Usage:
    python exp4_scalability.py --max-samples 100
"""

import argparse
import os
import sys
import time

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


class ScalabilityExperiment(ExperimentBase):
    """Experiment 4: Scalability Study"""

    def run(self):
        """Run scalability study experiment"""
        self.print_header("EXPERIMENT 4: SCALABILITY STUDY")

        model_configs = [
            {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "size": "1.1B",
                "max_samples": min(50, self.max_samples),
            },
            {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "size": "1.5B",
                "max_samples": min(50, self.max_samples),
            },
            # Add more models if GPU memory allows
            # {
            #     "name": "Qwen/Qwen2.5-3B-Instruct",
            #     "size": "3B",
            #     "max_samples": min(30, self.max_samples)
            # },
        ]

        results = []

        for model_config in model_configs:
            model_name = model_config["name"]
            model_size = model_config["size"]
            max_samples = model_config["max_samples"]

            print(f"\n{'═'*80}")
            print(f"Testing model: {model_name} ({model_size})")
            print(f"{'═'*80}")

            try:
                # Load model
                print(f"Loading tokenizer and model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                ).to(self.device)
                model.tokenizer = tokenizer
                print("✓ Model loaded\n")

                # Test with standard config
                config = QKVCommConfig(
                    mode="hybrid",
                    quantization_enabled=True,
                    calibration_enabled=True,
                    target_bits=6.0,
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
                    output_dir=str(self.output_dir / f"exp4_model_{model_size}"),
                    enable_baseline=True,
                )

                start_time = time.time()
                dataset_results = benchmark.run_benchmark(
                    dataset_names=["squad", "narrativeqa"],
                    max_samples=max_samples,
                )
                total_time = time.time() - start_time

                # Aggregate results
                for dataset_name, dataset_metrics in dataset_results.items():
                    successful = [r for r in dataset_metrics if r.get("success", False)]
                    if successful:
                        results.append(
                            {
                                "model_size": model_size,
                                "model_name": model_name,
                                "dataset": dataset_name,
                                "num_samples": len(successful),
                                "contextual_relevance": np.mean(
                                    [r["contextual_relevance"] for r in successful]
                                ),
                                "answer_completeness": np.mean(
                                    [r["answer_completeness"] for r in successful]
                                ),
                                "compression_ratio": np.mean(
                                    [r["compression_ratio"] for r in successful]
                                ),
                                "total_bandwidth_saved_mb": np.sum(
                                    [r["bits_saved"] for r in successful]
                                )
                                / 1e6,  # Convert bits to MB
                                "avg_inference_time": np.mean(
                                    [r["inference_time"] for r in successful]
                                ),
                                "total_time": total_time,
                                "compression_quality": np.mean(
                                    [
                                        r.get("compression_quality_score", 1.0)
                                        for r in successful
                                    ]
                                ),
                            }
                        )

                # Free memory
                del model
                del tokenizer
                torch.cuda.empty_cache() if self.device == "cuda" else None

            except Exception as e:
                print(f"❌ Error with model {model_name}: {e}")
                continue

        # Save results
        if results:
            df = pd.DataFrame(results)
            output_file = self.output_dir / "experiment_4_scalability.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to {output_file}")

            # Summary
            print("\n📊 Summary:")
            print(
                df.groupby("model_size")
                .agg(
                    {
                        "contextual_relevance": "mean",
                        "compression_ratio": "mean",
                        "avg_inference_time": "mean",
                    }
                )
                .round(4)
            )

            return df
        else:
            print("❌ No results to save")
            return None


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Scalability Study")
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
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Run experiment
    experiment = ScalabilityExperiment(
        output_dir=args.output_dir,
        device=args.device,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )

    experiment.run()


if __name__ == "__main__":
    main()
