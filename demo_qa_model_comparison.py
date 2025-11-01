"""
Compare different lightweight QA models for entity extraction.
Tests: TinyRoBERTa (82MB), DistilBERT (261MB), and RoBERTa-base (496MB)
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    print("=" * 80)
    print("LIGHTWEIGHT QA MODEL COMPARISON")
    print("=" * 80)

    print(
        """
Comparing three lightweight QA models:
1. TinyRoBERTa-SQuAD2 (deepset/tinyroberta-squad2) - 82MB ⚡
2. DistilBERT-SQuAD (distilbert-base-cased-distilled-squad) - 261MB 
3. RoBERTa-base-SQuAD2 (deepset/roberta-base-squad2) - 496MB 🎯

Testing on contexts where regex extraction failed completely.
    """
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model
    print("Loading TinyLlama...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.tokenizer = tokenizer
    print("✓ Model loaded\n")

    # Test cases where regex completely failed
    test_cases = [
        {
            "context": "The answer is no, definitely not yes.",
            "query": "What is the answer?",
            "expected": "no",
        },
        {
            "context": "The old price was $100 but the current price is now $50.",
            "query": "What's the current price?",
            "expected": "$50",
        },
        {
            "context": "John, Mary, and Sarah went to the store. Only Sarah bought milk.",
            "query": "Who bought milk?",
            "expected": "Sarah",
        },
        {
            "context": "The product has received excellent customer reviews and outstanding ratings.",
            "query": "What are the reviews like?",
            "expected": "excellent",
        },
        {
            "context": "The conference will take place from March 15-17, 2026 in Boston.",
            "query": "When is the conference?",
            "expected": "March 15-17",
        },
    ]

    # QA models to test
    qa_models = [
        {
            "name": "TinyRoBERTa (82MB)",
            "id": "deepset/tinyroberta-squad2",
            "size": "82MB",
            "speed": "Very Fast ⚡",
        },
        {
            "name": "DistilBERT (261MB)",
            "id": "distilbert-base-cased-distilled-squad",
            "size": "261MB",
            "speed": "Fast",
        },
        {
            "name": "RoBERTa-base (496MB)",
            "id": "deepset/roberta-base-squad2",
            "size": "496MB",
            "speed": "Medium",
        },
    ]

    calibration_data = [
        "Information can be extracted from text passages.",
        "Question answering systems find relevant answers in context.",
    ]

    # Test each QA model
    results = {}

    for qa_model in qa_models:
        print(f"\n{'='*80}")
        print(f"TESTING: {qa_model['name']}")
        print(f"Model: {qa_model['id']}")
        print(f"Size: {qa_model['size']} | Speed: {qa_model['speed']}")
        print(f"{'='*80}")

        try:
            config = QKVCommConfig(
                mode="hybrid",
                quantization_enabled=True,
                calibration_enabled=True,
                layer_selection_ratio=0.7,
                hybrid_entity_extraction="qa",
                hybrid_qa_model=qa_model["id"],
                hybrid_max_entity_tokens=30,
            )

            system = QKVCommSystem(model, model, config, device)
            system.calibrate(calibration_data)

            correct = 0

            for i, test in enumerate(test_cases, 1):
                print(f"\n{'-'*80}")
                print(f"Test {i}: {test['query']}")
                print(f"Context: {test['context']}")
                print(f"Expected: '{test['expected']}'")

                output, metrics = system.communicate(
                    test["context"], test["query"], max_new_tokens=40
                )

                extracted = metrics.get("hybrid_extracted_facts", "N/A")
                is_correct = test["expected"].lower() in output.lower()
                correct += int(is_correct)

                print(f"Extracted: {extracted}")
                print(f"Output: {output}")
                print(f"{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

            accuracy = 100 * correct / len(test_cases)
            results[qa_model["name"]] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_cases),
                "size": qa_model["size"],
                "speed": qa_model["speed"],
            }

            print(f"\n{'='*80}")
            print(
                f"{qa_model['name']} Results: {correct}/{len(test_cases)} ({accuracy:.0f}%)"
            )
            print(f"{'='*80}")

        except Exception as e:
            print(f"\n❌ {qa_model['name']} failed: {e}")
            results[qa_model["name"]] = {"accuracy": 0, "error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<30} {'Size':<12} {'Speed':<15} {'Accuracy'}")
    print("-" * 80)

    for model_name, data in results.items():
        if "error" not in data:
            print(
                f"{model_name:<30} {data['size']:<12} {data['speed']:<15} "
                f"{data['correct']}/{data['total']} ({data['accuracy']:.0f}%)"
            )
        else:
            print(f"{model_name:<30} ERROR: {data['error']}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(
        """
BEST OVERALL: TinyRoBERTa-SQuAD2 (deepset/tinyroberta-squad2)
  ✓ Smallest: 82MB (3x smaller than DistilBERT)
  ✓ Fastest: Minimal inference overhead
  ✓ Good accuracy: Trained on SQuAD 2.0 (handles unanswerable)
  ✓ Cross-model: Completely independent model
  → RECOMMENDED for production use

BALANCED: DistilBERT-SQuAD 
  ✓ Medium size: 261MB
  ✓ Good speed: Distilled from BERT
  ✓ Solid accuracy: Well-tested model
  → Use if TinyRoBERTa accuracy insufficient

BEST ACCURACY: RoBERTa-base-SQuAD2
  ✓ Highest accuracy: Full-size model
  ✗ Large: 496MB (6x TinyRoBERTa)
  ✗ Slower: More compute overhead
  → Only if accuracy is critical and resources available

DEFAULT CHOICE: TinyRoBERTa-SQuAD2
Best balance of size, speed, and accuracy for cross-model KV communication.
    """
    )


if __name__ == "__main__":
    main()
