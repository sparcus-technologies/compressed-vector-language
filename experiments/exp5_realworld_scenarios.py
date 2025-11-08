"""
Experiment 5: Real-world Scenarios

Tests Q-KVComm in practical multi-agent scenarios:
- Conversational QA (with conversation history caching)
- Multi-hop reasoning (complex question answering)

Usage:
    python exp5_realworld_scenarios.py --max-samples 100
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


class RealWorldScenariosExperiment(ExperimentBase):
    """Experiment 5: Real-world Scenarios"""

    def run(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Run real-world scenarios experiment"""
        self.print_header("EXPERIMENT 5: REAL-WORLD SCENARIOS")
        print("ℹ️  This experiment tests conversational and multi-hop scenarios\n")

        results = []

        # Load model
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.tokenizer = tokenizer
        print("✓ Model loaded\n")

        # Scenario 1: Conversational QA (CoQA)
        self.print_section("Scenario 1: Conversational Question Answering")

        config = QKVCommConfig(
            mode="hybrid",
            quantization_enabled=True,
            calibration_enabled=True,
            target_bits=6.0,
            extraction_method="hybrid",  # Hybrid works well for conversation
            extraction_cache_enabled=True,  # Cache for conversation history
            max_memory_mb=1024,
        )

        # Create Q-KVComm system
        qkvcomm_system = QKVCommSystem(
            sender_model=model,
            receiver_model=model,
            config=config,
            device=self.device,
        )

        benchmark = BenchmarkSuite(
            qkvcomm_system=qkvcomm_system,
            output_dir=str(self.output_dir / "exp5_scenario_conversational"),
            enable_baseline=True,
        )

        dataset_results = benchmark.run_benchmark(
            dataset_names=["coqa"],
            max_samples=min(50, self.max_samples),
        )

        for dataset_name, dataset_metrics in dataset_results.items():
            successful = [r for r in dataset_metrics if r.get("success", False)]
            if successful:
                results.append(
                    {
                        "scenario": "conversational_qa",
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
                        "bandwidth_saved_mb": np.sum(
                            [r["bits_saved"] for r in successful]
                        )
                        / 1e6,  # Convert bits to MB
                        "cache_hit_rate": np.mean(
                            [
                                r.get("memory_stats", {}).get("hit_rate", 0)
                                for r in successful
                            ]
                        ),
                    }
                )

        # Scenario 2: Multi-hop Reasoning (HotpotQA)
        self.print_section("Scenario 2: Multi-hop Reasoning")

        config = QKVCommConfig(
            mode="hybrid",
            quantization_enabled=True,
            calibration_enabled=True,
            target_bits=8.0,  # Higher precision for complex reasoning
            layer_selection_ratio=0.9,  # More layers for reasoning
            extraction_method="hybrid",
            max_memory_mb=1024,
        )

        # Create Q-KVComm system
        qkvcomm_system = QKVCommSystem(
            sender_model=model,
            receiver_model=model,
            config=config,
            device=self.device,
        )

        benchmark = BenchmarkSuite(
            qkvcomm_system=qkvcomm_system,
            output_dir=str(self.output_dir / "exp5_scenario_multihop"),
            enable_baseline=True,
        )

        dataset_results = benchmark.run_benchmark(
            dataset_names=["hotpot_qa"],
            max_samples=min(50, self.max_samples),
        )

        for dataset_name, dataset_metrics in dataset_results.items():
            successful = [r for r in dataset_metrics if r.get("success", False)]
            if successful:
                results.append(
                    {
                        "scenario": "multi_hop_reasoning",
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
                        "bandwidth_saved_mb": np.sum(
                            [r["bits_saved"] for r in successful]
                        )
                        / 1e6,  # Convert bits to MB
                        "avg_inference_time": np.mean(
                            [r["inference_time"] for r in successful]
                        ),
                    }
                )

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "experiment_5_realworld_scenarios.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Summary
        print("\n📊 Summary:")
        print(
            df.groupby("scenario")
            .agg(
                {
                    "contextual_relevance": "mean",
                    "answer_completeness": "mean",
                    "compression_ratio": "mean",
                    "bandwidth_saved_mb": "sum",
                }
            )
            .round(4)
        )

        return df


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Real-world Scenarios")
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
    experiment = RealWorldScenariosExperiment(
        output_dir=args.output_dir,
        device=args.device,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )

    experiment.run(model_name=args.model)


if __name__ == "__main__":
    main()
