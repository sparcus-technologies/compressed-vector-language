"""Run benchmark with Ollama local models - 5 datasets"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmark_suite import BenchmarkSuite
from transformers import AutoModelForCausalLM, AutoTokenizer

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    print("=" * 80)
    print("Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK")
    print("5 Datasets: SQuAD, HotpotQA, Natural Questions, CoQA, NarrativeQA")
    print("=" * 80)

    # Force CPU usage only
    device = "cpu"

    # Use instruction-tuned model for better agentic communication
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"\nModel: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading sender model...")
    sender = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    sender = sender.to(device)
    sender.tokenizer = tokenizer

    print("Loading receiver model...")
    receiver = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    receiver = receiver.to(device)
    receiver.tokenizer = tokenizer

    # Configure Q-KVComm for optimal agentic communication
    # Balanced configuration: better quality while maintaining good compression
    config = QKVCommConfig(
        mode="full",
        target_bits=8.0,  # Increased from 6.0 for better quality preservation
        layer_selection_ratio=0.8,  # Increased from 0.6 to preserve more semantic info
        quantization_enabled=True,
        calibration_enabled=True,
        profiling_samples=50,  # Increased for better calibration
        calibration_samples=30,  # Increased for better calibration
    )

    qkvcomm = QKVCommSystem(sender, receiver, config, device)

    # Create benchmark suite
    # Set enable_baseline=True to compare compressed vs uncompressed (slower but insightful)
    enable_baseline = True

    benchmark = BenchmarkSuite(
        qkvcomm_system=qkvcomm,
        output_dir="benchmark_results",
        enable_baseline=enable_baseline,
    )

    print(f"\nRunning benchmark across 5 datasets...")
    if enable_baseline:
        print("⚠️  Baseline comparison enabled - evaluation will take 2x longer")
        print(
            "   This will provide compression quality and semantic preservation metrics\n"
        )

    # Run benchmark with 5 diverse datasets
    benchmark.run_benchmark(
        dataset_names=[
            "squad",  # Extractive QA
            "hotpot_qa",  # Multi-hop reasoning
            "natural_questions",  # Open domain QA
            "coqa",  # Conversational QA
            "narrativeqa",  # Long-form reading comprehension
        ],
        max_samples=5,  # samples per dataset
        max_new_tokens=50,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - Results saved to benchmark_results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
