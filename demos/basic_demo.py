"""
Basic Q-KVComm Demonstration

This demo shows the fundamental capabilities of Q-KVComm:
- KV cache compression and transfer
- Quantization and calibration
- Simple question-answering scenario
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load models
    print("Loading models...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sender = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sender.tokenizer = tokenizer

    receiver = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    receiver.tokenizer = tokenizer

    # Setup Q-KVComm
    config = QKVCommConfig(
        mode="full",
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
    )

    qkvcomm = QKVCommSystem(sender, receiver, config, device)

    # Calibration
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Space exploration reveals cosmic mysteries.",
    ]

    qkvcomm.calibrate(calibration_data)

    # Demo communication
    context = "Artificial intelligence includes machine learning and deep learning."
    query = "What does AI include?"

    print(f"\nContext: {context}")
    print(f"Query: {query}")
    print("\nGenerating response...")

    output, metrics = qkvcomm.communicate(context, query, max_new_tokens=30)

    print(f"\nOutput: {output}")
    print(f"\nMetrics:")
    print(f"  Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
    print(f"  Layers Transmitted: {metrics['num_layers_transmitted']}")
    print(f"  Communication Saved: {(1 - 1/metrics['avg_compression_ratio'])*100:.1f}%")


if __name__ == "__main__":
    main()
