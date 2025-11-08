"""
Quantization Demo - Understanding Compression Levels

This demo shows how different quantization bit-widths affect:
- Compression ratio
- Answer quality
- Communication bandwidth saved

Try different compression levels to find the sweet spot for your use case!
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def test_quantization_level(
    sender, receiver, device, target_bits, context, query, calibration_data
):
    """Test a specific quantization level"""
    # Adjust max_bits if target_bits exceeds the default max
    max_bits = max(8, int(target_bits))

    config = QKVCommConfig(
        mode="full",
        target_bits=target_bits,
        min_bits=4,
        max_bits=max_bits,
        quantization_enabled=True,
        calibration_enabled=True,
    )

    qkvcomm = QKVCommSystem(sender, receiver, config, device)
    qkvcomm.calibrate(calibration_data)

    output, metrics = qkvcomm.communicate(context, query, max_new_tokens=40)

    return output, metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Quantization Demo")
    print(f"Device: {device}\n")

    # Load models
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sender = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sender.tokenizer = tokenizer

    receiver = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    receiver.tokenizer = tokenizer
    print("✓ Model loaded\n")

    # Calibration data
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Space exploration reveals cosmic mysteries.",
        "Quantum computing promises revolutionary advances.",
    ]

    # Test scenario
    context = (
        "Python is a high-level, interpreted programming language known for its "
        "simplicity and readability. Created by Guido van Rossum in 1991, Python "
        "emphasizes code readability with significant whitespace. It supports multiple "
        "programming paradigms including procedural, object-oriented, and functional "
        "programming. Python has a comprehensive standard library and a vast ecosystem "
        "of third-party packages."
    )
    query = "What are the main features of Python?"

    print("=" * 80)
    print("TEST SCENARIO")
    print("=" * 80)
    print(f"Context: {context[:100]}...")
    print(f"Query: {query}\n")

    # Test different quantization levels
    quantization_levels = [
        (4.0, "Aggressive (4-bit)", "Maximum compression, lower quality"),
        (6.0, "Balanced (6-bit)", "Good compression, good quality"),
        (8.0, "Conservative (8-bit)", "Moderate compression, high quality"),
        (16.0, "Minimal (16-bit)", "Low compression, maximum quality"),
    ]

    results = []

    for bits, name, description in quantization_levels:
        print("=" * 80)
        print(f"{name} - {description}")
        print("=" * 80)

        output, metrics = test_quantization_level(
            sender, receiver, device, bits, context, query, calibration_data
        )

        print(f"Answer: {output}\n")
        print(f"📊 Metrics:")
        print(f"  • Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
        print(f"  • Layers Transmitted: {metrics['num_layers_transmitted']}")
        print(f"  • Bits per Value: {bits}")
        print(
            f"  • Bandwidth Saved: {(1 - 1/metrics['avg_compression_ratio'])*100:.1f}%"
        )
        print()

        results.append(
            {
                "name": name,
                "bits": bits,
                "compression": metrics["avg_compression_ratio"],
                "answer": output,
            }
        )

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n📈 Compression vs Quality Trade-off:\n")
    print(f"{'Quantization Level':<25} {'Bits':<8} {'Compression':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<25} {r['bits']:<8.1f} {r['compression']:<15.2f}x")

    print("\n💡 Key Insights:")
    print("  • Lower bits = Higher compression but may affect quality")
    print("  • 6-bit quantization typically offers the best balance")
    print("  • 4-bit is great for bandwidth-constrained scenarios")
    print("  • 8-bit+ ensures maximum quality preservation")

    print("\n🎯 Recommendation:")
    print("  • For most use cases: Use 6-bit quantization")
    print("  • For critical applications: Use 8-bit")
    print("  • For extreme compression: Use 4-bit")


if __name__ == "__main__":
    main()
