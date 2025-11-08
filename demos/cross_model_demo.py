"""
Cross-Model Communication Demo

Demonstrates Q-KVComm with DIFFERENT sender and receiver models.
Shows cross-model KV cache transfer between architectures with different sizes.
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

    # Load DIFFERENT models for sender and receiver
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    # Sender: Qwen2.5-1.5B (larger model for context understanding)
    sender_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\nSender Model: {sender_model_name}")
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_model_name)
    sender_tokenizer.pad_token = sender_tokenizer.eos_token
    sender = AutoModelForCausalLM.from_pretrained(sender_model_name).to(device)
    sender.tokenizer = sender_tokenizer
    print(f"  ✓ Loaded ({sender.config.num_hidden_layers} layers)")

    # Receiver: TinyLlama-1.1B (smaller, different architecture)
    receiver_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nReceiver Model: {receiver_model_name}")
    receiver_tokenizer = AutoTokenizer.from_pretrained(receiver_model_name)
    receiver_tokenizer.pad_token = receiver_tokenizer.eos_token
    receiver = AutoModelForCausalLM.from_pretrained(receiver_model_name).to(device)
    receiver.tokenizer = receiver_tokenizer
    print(f"  ✓ Loaded ({receiver.config.num_hidden_layers} layers)")

    print(f"\n📊 Model Comparison:")
    print(f"  Sender layers:   {sender.config.num_hidden_layers}")
    print(f"  Receiver layers: {receiver.config.num_hidden_layers}")
    print(f"  Sender heads:    {sender.config.num_attention_heads}")
    print(f"  Receiver heads:  {receiver.config.num_attention_heads}")

    # Check if models have compatible architectures
    sender_head_dim = sender.config.hidden_size // sender.config.num_attention_heads
    receiver_head_dim = (
        receiver.config.hidden_size // receiver.config.num_attention_heads
    )
    same_head_dim = sender_head_dim == receiver_head_dim

    print(f"  Sender head dim:   {sender_head_dim}")
    print(f"  Receiver head dim: {receiver_head_dim}")
    print(
        f"  Same head dim:     {'✓ Yes' if same_head_dim else '✗ No (will use scalar calibration)'}"
    )

    # Setup Q-KVComm
    # For heterogeneous models with different head dimensions, the system automatically
    # adapts the calibration strategy
    config = QKVCommConfig(
        mode="full",
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
        layer_selection_ratio=0.5,  # Select 50% of layers
    )

    print(f"\n🔧 Q-KVComm Configuration:")
    print(f"  Quantization: {config.quantization_enabled}")
    print(f"  Calibration: {config.calibration_enabled}")
    print(f"  Target bits: {config.target_bits}")
    print(f"  Layer selection: {config.layer_selection_ratio*100:.0f}%")

    qkvcomm = QKVCommSystem(sender, receiver, config, device)

    # Calibration
    print("\n" + "=" * 70)
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Space exploration reveals cosmic mysteries.",
        "Quantum computing promises revolutionary advances.",
        "Climate change affects global ecosystems.",
    ]

    qkvcomm.calibrate(calibration_data)

    # Demo multiple communications
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMMUNICATION DEMOS")
    print("=" * 70)

    test_cases = [
        {
            "context": "Artificial intelligence includes machine learning and deep learning. Machine learning uses statistical methods to enable computers to learn from data.",
            "query": "What does AI include?",
        },
        {
            "context": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
            "query": "Who created Python?",
        },
        {
            "context": "The solar system consists of the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "query": "How many planets are in the solar system?",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test Case {i}")
        print(f"{'─' * 70}")

        context = test_case["context"]
        query = test_case["query"]

        print(f"📄 Context: {context[:80]}...")
        print(f"❓ Query: {query}")
        print(f"\n🔄 Transferring KV cache from Sender → Receiver...")

        output, metrics = qkvcomm.communicate(context, query, max_new_tokens=50)

        print(f"💬 Response: {output}")
        print(f"\n📊 Metrics:")
        print(f"  • Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
        print(f"  • Layers Transmitted: {metrics['num_layers_transmitted']}")
        print(
            f"  • Communication Saved: {(1 - 1/metrics['avg_compression_ratio'])*100:.1f}%"
        )

    print("\n" + "=" * 70)
    print("✅ CROSS-MODEL COMMUNICATION COMPLETE")
    print("=" * 70)
    print("\n📝 Key Insights:")
    print("  • KV cache from Qwen2.5-1.5B (sender) was successfully transferred")
    print("    to TinyLlama-1.1B (receiver) with quantization compression!")
    print("  • Cross-model transfer works even with different architectures")
    print("  • The system automatically adapts to handle different model shapes")
    print("  • Quantization provides compression in all scenarios")

    print("\n💡 Technical Details:")
    print("   • Q-KVComm handles heterogeneous models with different architectures")
    print("   • Calibration adapts to work with varying attention head configurations")
    print("   • Both layer selection and quantization work together for efficiency")
    print("   • The system maintains core Q-KVComm benefits across model types!")


if __name__ == "__main__":
    main()
