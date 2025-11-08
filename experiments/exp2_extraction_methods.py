"""
Experiment 2: Extraction Method Comparison

Ablation study comparing simple, YAKE, SpaCy, and hybrid extraction methods.
Shows which extraction method provides the best balance of quality and efficiency.

Usage:
    python exp2_extraction_methods.py --max-samples 100
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


class ExtractionMethodExperiment(ExperimentBase):
    """Experiment 2: Extraction Method Comparison"""

    def run(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Run extraction method comparison experiment"""
        self.print_header("EXPERIMENT 2: EXTRACTION METHOD COMPARISON")

        extraction_methods = ["simple", "yake", "spacy_ner", "hybrid"]
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

        for method in extraction_methods:
            self.print_section(f"Testing extraction method: {method.upper()}")

            config = QKVCommConfig(
                mode="hybrid",
                quantization_enabled=True,
                calibration_enabled=True,
                target_bits=6.0,
                extraction_method=method,
                extraction_max_tokens=30,
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
                output_dir=str(self.output_dir / f"exp2_method_{method}"),
                enable_baseline=True,
            )

            start_time = time.time()
            dataset_results = benchmark.run_benchmark(
                dataset_names=["squad", "hotpot_qa", "coqa", "narrativeqa"],
                max_samples=min(50, self.max_samples),
            )
            total_time = time.time() - start_time

            # Aggregate results
            for dataset_name, dataset_metrics in dataset_results.items():
                successful = [r for r in dataset_metrics if r.get("success", False)]
                if successful:
                    results.append(
                        {
                            "extraction_method": method,
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
                            "num_facts_extracted": np.mean(
                                [r.get("num_facts_extracted", 0) for r in successful]
                            ),
                            "extraction_cache_hits": np.sum(
                                [
                                    r.get("extraction_cache_hit", False)
                                    for r in successful
                                ]
                            ),
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

        # Save results
        df = pd.DataFrame(results)
        output_file = self.output_dir / "experiment_2_extraction_methods.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Summary
        print("\n📊 Summary:")
        print(
            df.groupby("extraction_method")
            .agg(
                {
                    "contextual_relevance": "mean",
                    "num_facts_extracted": "mean",
                    "avg_inference_time": "mean",
                    "compression_quality": "mean",
                }
            )
            .round(4)
        )

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Extraction Method Comparison"
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
    experiment = ExtractionMethodExperiment(
        output_dir=args.output_dir,
        device=args.device,
        max_samples_per_dataset=args.max_samples,
        seed=args.seed,
    )

    experiment.run(model_name=args.model)


if __name__ == "__main__":
    main()
