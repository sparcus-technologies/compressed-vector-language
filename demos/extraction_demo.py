"""
Adaptive Extraction Demo - Smart Information Extraction

This demo shows Q-KVComm's adaptive extraction capabilities:
- Extracting key facts from long contexts
- Different extraction methods (YAKE, SpaCy NER, hybrid)
- Selective KV cache transmission based on relevance
- Combining extraction with compression for maximum efficiency

Q-KVComm can intelligently extract important information and transmit
only the relevant KV cache entries - not everything!
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def test_extraction_method(
    sender, receiver, device, extraction_method, context, query, calibration_data
):
    """Test a specific extraction method"""
    config = QKVCommConfig(
        mode="hybrid",  # Use hybrid mode for extraction + compression
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
        extraction_method=extraction_method,
        extraction_max_tokens=50,  # Extract up to 50 key tokens
    )

    qkvcomm = QKVCommSystem(sender, receiver, config, device)
    qkvcomm.calibrate(calibration_data)

    output, metrics = qkvcomm.communicate(context, query, max_new_tokens=50)

    return output, metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Adaptive Extraction Demo")
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
    ]

    # Long context with lots of information (extraction will help!)
    context = """
    The Amazon rainforest, often called the "lungs of the Earth," is the world's largest 
    tropical rainforest, spanning approximately 5.5 million square kilometers across nine 
    South American countries. Brazil contains about 60% of the rainforest. The Amazon is 
    home to an estimated 390 billion individual trees divided into 16,000 species. It 
    produces about 20% of the world's oxygen and plays a crucial role in regulating global 
    climate patterns. The rainforest supports incredible biodiversity, with over 40,000 
    plant species, 1,300 bird species, 3,000 types of fish, 430 mammals, and an estimated 
    2.5 million insect species. Indigenous peoples have lived in the Amazon for at least 
    11,000 years, with around 400-500 indigenous tribes currently residing there. 
    Unfortunately, deforestation is a major threat, with an area roughly the size of a 
    football field cleared every single minute. Between 2000 and 2020, approximately 10% 
    of the Amazon was lost to deforestation, primarily due to cattle ranching, agriculture, 
    and logging. Climate change and fires are accelerating this destruction. Scientists 
    warn that if deforestation continues at the current rate, the Amazon could reach a 
    tipping point where it transforms from rainforest to savanna, which would have 
    catastrophic consequences for global climate and biodiversity.
    """

    # Test different extraction methods
    extraction_methods = [
        ("simple", "Simple keyword extraction", "Basic frequency-based extraction"),
        ("yake", "YAKE algorithm", "Statistical keyword extraction"),
        ("spacy_ner", "SpaCy NER", "Named entity recognition"),
        ("hybrid", "Hybrid approach", "Combines multiple methods"),
    ]

    queries = [
        "How much of the world's oxygen does the Amazon produce?",
        "What are the main threats to the Amazon rainforest?",
        "How many tree species are in the Amazon?",
    ]

    print("=" * 80)
    print("LONG CONTEXT SCENARIO")
    print("=" * 80)
    print(f"Context length: {len(context)} characters, {len(context.split())} words")
    print(f"Context preview: {context[:150]}...\n")

    # Test each query with different extraction methods
    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"QUERY {i}: {query}")
        print("=" * 80)

        for method, name, description in extraction_methods:
            print(f"\n{'─' * 80}")
            print(f"Extraction Method: {name}")
            print(f"Description: {description}")
            print(f"{'─' * 80}")

            try:
                output, metrics = test_extraction_method(
                    sender, receiver, device, method, context, query, calibration_data
                )

                print(f"\n💬 Answer: {output}")
                print(f"\n📊 Metrics:")
                print(f"  • Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
                print(
                    f"  • Layers Transmitted: {metrics['num_layers_transmitted']}/{metrics.get('total_layers', 'N/A')}"
                )

                # Check if extraction stats are available
                extraction_stats = metrics.get("extraction_stats", {})
                if extraction_stats:
                    print(
                        f"  • Facts Extracted: {extraction_stats.get('num_facts', 'N/A')}"
                    )
                    print(
                        f"  • Extraction Method: {extraction_stats.get('method', method)}"
                    )

                print(
                    f"  • Bandwidth Saved: {(1 - 1/metrics['avg_compression_ratio'])*100:.1f}%"
                )

            except Exception as e:
                print(f"  ⚠️  Error with {method}: {str(e)[:100]}")
                print(f"  This method may require additional dependencies")

    # Demonstrate no extraction vs with extraction
    print("\n\n" + "=" * 80)
    print("COMPARISON: NO EXTRACTION vs WITH EXTRACTION")
    print("=" * 80)

    test_query = "What percentage of the Amazon has been lost to deforestation?"
    print(f"\nQuery: {test_query}\n")

    # Without extraction (baseline mode)
    print("─" * 80)
    print("Mode 1: BASELINE (No Extraction, No Compression)")
    print("─" * 80)
    config_baseline = QKVCommConfig(
        mode="baseline",
        quantization_enabled=False,
        calibration_enabled=True,
    )
    qkvcomm_baseline = QKVCommSystem(sender, receiver, config_baseline, device)
    qkvcomm_baseline.calibrate(calibration_data)
    output_baseline, metrics_baseline = qkvcomm_baseline.communicate(
        context, test_query, max_new_tokens=50
    )
    print(f"Answer: {output_baseline}")
    print(f"Compression: {metrics_baseline['avg_compression_ratio']:.2f}x")
    print(f"Layers: {metrics_baseline['num_layers_transmitted']}")

    # With compression only (no extraction)
    print("\n" + "─" * 80)
    print("Mode 2: COMPRESSION ONLY (No Extraction)")
    print("─" * 80)
    config_compress = QKVCommConfig(
        mode="quantization_only",
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
    )
    qkvcomm_compress = QKVCommSystem(sender, receiver, config_compress, device)
    qkvcomm_compress.calibrate(calibration_data)
    output_compress, metrics_compress = qkvcomm_compress.communicate(
        context, test_query, max_new_tokens=50
    )
    print(f"Answer: {output_compress}")
    print(f"Compression: {metrics_compress['avg_compression_ratio']:.2f}x")
    print(f"Layers: {metrics_compress['num_layers_transmitted']}")

    # With extraction + compression (full Q-KVComm)
    print("\n" + "─" * 80)
    print("Mode 3: FULL Q-KVCOMM (Extraction + Compression)")
    print("─" * 80)
    config_full = QKVCommConfig(
        mode="hybrid",
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
        extraction_method="hybrid",
        extraction_max_tokens=50,
    )
    qkvcomm_full = QKVCommSystem(sender, receiver, config_full, device)
    qkvcomm_full.calibrate(calibration_data)
    output_full, metrics_full = qkvcomm_full.communicate(
        context, test_query, max_new_tokens=50
    )
    print(f"Answer: {output_full}")
    print(f"Compression: {metrics_full['avg_compression_ratio']:.2f}x")
    print(f"Layers: {metrics_full['num_layers_transmitted']}")

    extraction_stats = metrics_full.get("extraction_stats", {})
    if extraction_stats:
        print(f"Facts Extracted: {extraction_stats.get('num_facts', 'N/A')}")

    # Summary
    print("\n\n" + "=" * 80)
    print("✅ ADAPTIVE EXTRACTION DEMO COMPLETE")
    print("=" * 80)

    print("\n🎯 Key Takeaways:")
    print("  • Extraction identifies and focuses on relevant information")
    print("  • Different extraction methods work better for different tasks:")
    print("    - Simple: Fast, good for keyword-based queries")
    print("    - YAKE: Statistical, great for factual questions")
    print("    - SpaCy NER: Best for entity-focused queries (people, places, etc.)")
    print("    - Hybrid: Combines strengths of multiple methods")
    print("  • Combining extraction with compression maximizes efficiency")
    print("  • Long contexts benefit most from extraction!")

    print("\n💡 When to Use Extraction:")
    print("  • Long documents with specific queries")
    print("  • Question answering tasks")
    print("  • Information retrieval scenarios")
    print("  • Bandwidth-constrained environments")


if __name__ == "__main__":
    main()
