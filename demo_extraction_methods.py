"""
Compare different extraction methods: simple, attention, QA, and summarization.
Shows which is most reliable for cross-model communication.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    print("=" * 80)
    print("EXTRACTION METHOD COMPARISON")
    print("=" * 80)
    print("\nComparing: Simple (regex), QA model, and Summarization\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.tokenizer = tokenizer
    print("✓ Model loaded\n")

    # Test cases where regex fails
    test_cases = [
        {
            "context": "The answer is no, not yes.",
            "query": "What is the answer?",
            "expected": "no",
        },
        {
            "context": "The old price was $100 but now it's $50.",
            "query": "What's the current price?",
            "expected": "50",
        },
        {
            "context": "John, Mary, and Sarah went shopping. Sarah bought milk.",
            "query": "Who bought milk?",
            "expected": "Sarah",
        },
        {
            "context": "The product has excellent reviews and high quality.",
            "query": "What are the reviews like?",
            "expected": "excellent",
        },
    ]

    # Calibration data
    calibration_data = [
        "Machine learning enables pattern recognition.",
        "Natural language processing understands text.",
    ]

    # Test each method
    methods = ["simple", "qa", "summarization"]

    for method in methods:
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")

        try:
            config = QKVCommConfig(
                mode="hybrid",
                quantization_enabled=True,
                calibration_enabled=True,
                layer_selection_ratio=0.7,
                hybrid_entity_extraction=method,
                hybrid_max_entity_tokens=20,
            )

            system = QKVCommSystem(model, model, config, device)
            system.calibrate(calibration_data)

            correct = 0
            total = len(test_cases)

            for i, test in enumerate(test_cases, 1):
                print(f"\n{'-'*80}")
                print(f"Test {i}: {test['query']}")
                print(f"Context: {test['context']}")
                print(f"Expected: '{test['expected']}'")

                output, metrics = system.communicate(
                    test["context"], test["query"], max_new_tokens=30
                )

                is_correct = test["expected"].lower() in output.lower()
                correct += int(is_correct)

                print(f"Extracted: {metrics.get('hybrid_extracted_facts', 'N/A')}")
                print(f"Output: {output}")
                print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

            accuracy = 100 * correct / total
            print(f"\n{'='*80}")
            print(f"{method.upper()} ACCURACY: {correct}/{total} ({accuracy:.0f}%)")
            print(f"{'='*80}")

        except Exception as e:
            print(f"\n❌ {method.upper()} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(
        """
SIMPLE (Regex):
  ✓ Fast, no extra models
  ✓ Works cross-model
  ✗ Unreliable (case-sensitive, misses context)
  → Use when: Speed matters, well-formatted text

QA Model:
  ✓ Query-aware extraction
  ✓ Works cross-model (separate QA model)
  ✓ More reliable than regex
  ✗ Extra model to load (DistilBERT ~250MB)
  → Use when: Need reliability, can afford extra model

SUMMARIZATION:
  ✓ Most accurate (uses sender model's understanding)
  ✓ Context-aware
  ✗ Not fully cross-model (uses sender model)
  ✗ Slower (requires generation)
  → Use when: Same model or accuracy is critical

BEST FOR CROSS-MODEL: QA method
- Separate model means no architectural dependency
- More reliable than regex
- Still efficient (~20 tokens)
    """
    )


if __name__ == "__main__":
    main()
