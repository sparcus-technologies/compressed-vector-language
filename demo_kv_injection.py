"""
Final Demo: Clear examples showing when KV injection works and when it doesn't.

This demonstrates the actual behavior of the Q-KVComm system:
- Technical success: KV cache IS being transferred and injected
- Practical challenge: Information transfer is lossy and inconsistent
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    print("=" * 80)
    print("KV CACHE INJECTION DEMO")
    print("=" * 80)
    print("\nThis demo shows how KV cache injection transfers information")
    print("between models WITHOUT sending the actual text.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model
    print("Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.tokenizer = tokenizer
    print("✓ Model loaded\n")

    # Setup systems
    print("Setting up communication systems...")

    # Baseline: Normal text-based communication
    baseline = QKVCommSystem(model, model, QKVCommConfig(mode="baseline"), device)
    baseline._is_calibrated = True

    # KV Transfer: Communication via KV cache injection
    kv_transfer = QKVCommSystem(
        model,
        model,
        QKVCommConfig(
            mode="full",
            quantization_enabled=True,
            calibration_enabled=True,
            layer_selection_ratio=0.7,
        ),
        device,
    )

    calibration_data = [
        "Artificial intelligence includes machine learning and neural networks.",
        "The capital of France is Paris, which is famous for the Eiffel Tower.",
        "Python is a popular programming language for data science.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    ]
    kv_transfer.calibrate(calibration_data)
    print("✓ Systems ready\n")

    # Examples
    examples = [
        {
            "name": "SUCCESS CASE",
            "context": "The capital of France is Paris. Paris has the Eiffel Tower.",
            "query": "What is the capital of France?",
            "why": "Likely in training data, strong semantic relationship",
        },
        {
            "name": "SUCCESS CASE",
            "context": "Python was created by Guido van Rossum in 1991.",
            "query": "Who created Python?",
            "why": "Well-known fact present in model's training data",
        },
        {
            "name": "PARTIAL SUCCESS",
            "context": "The speed of light is 299,792 km/s.",
            "query": "What is the speed of light?",
            "why": "Gets the concept right, may lose exact numbers",
        },
        {
            "name": "FAILURE CASE",
            "context": "Alice is 28 years old and works at Microsoft.",
            "query": "How old is Alice?",
            "why": "Specific novel information not in training data",
        },
        {
            "name": "FAILURE CASE",
            "context": "The meeting is at 2:30 PM in Room 305.",
            "query": "When is the meeting?",
            "why": "Specific details are poorly preserved",
        },
    ]

    for i, example in enumerate(examples, 1):
        print("=" * 80)
        print(f"EXAMPLE {i}: {example['name']}")
        print("=" * 80)
        print(f"\nContext (what sender knows):")
        print(f"  → {example['context']}")
        print(f"\nQuery (what receiver is asked):")
        print(f"  → {example['query']}")
        print(f"\nExpected behavior:")
        print(f"  → {example['why']}")

        # Baseline
        print(f"\n{'-' * 80}")
        print("METHOD 1: BASELINE (receiver gets full text)")
        print(f"{'-' * 80}")
        baseline_out, _ = baseline.communicate(
            example["context"], example["query"], max_new_tokens=50
        )
        print(f"Output: {baseline_out}")

        # KV Transfer
        print(f"\n{'-' * 80}")
        print("METHOD 2: KV INJECTION (receiver gets only KV cache)")
        print(f"{'-' * 80}")
        kv_out, metrics = kv_transfer.communicate(
            example["context"], example["query"], max_new_tokens=50
        )
        print(f"Output: {kv_out}")
        print(
            f"Compression: {metrics['avg_compression_ratio']:.1f}x "
            f"({metrics['num_layers_transmitted']} layers)"
        )

        print()

    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(
        """
✓ TECHNICAL SUCCESS:
  - KV cache extraction works
  - Quantization reduces size (2-3x compression)
  - KV injection into receiver works
  - System runs end-to-end

⚠ PRACTICAL CHALLENGES:
  - Information transfer is LOSSY
  - Works best for well-known facts
  - Fails for novel specific details
  - Numbers and names often lost
  
💡 WHY THIS HAPPENS:
  - KV cache stores attention patterns, not raw facts
  - Context is "compressed" into attention space
  - Receiver doesn't see actual tokens
  - Works like "semantic hints" rather than direct transfer
  
📊 WHEN IT WORKS:
  - Information already in model's training data
  - Strong semantic relationships
  - General concepts vs specific details
  - Continuation/style transfer tasks
  
❌ WHEN IT FAILS:
  - Novel information (names, dates, IDs)
  - Specific numeric values
  - Facts requiring exact recall
  - Information not in training distribution

🔬 RESEARCH INSIGHT:
  This demonstrates a fundamental limitation of KV cache communication:
  It's not a perfect information channel, but rather a "semantic compression"
  that works when the receiver can reconstruct information from hints.
"""
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
