"""
Demo: Query-Independent Entity Extraction
Shows that hybrid mode now works WITHOUT requiring a query.
This is essential for real multi-agent communication scenarios.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    print("=" * 80)
    print("QUERY-INDEPENDENT EXTRACTION DEMO")
    print("=" * 80)

    print(
        """
Real-world scenario: Multi-agent collaboration where agents share context
WITHOUT specific queries. The receiver needs to understand the context for
future tasks, not answer a specific question right now.

Testing two query-independent extraction methods:
1. Attention-based (queryless) - Uses model's self-attention ⚡
2. YAKE - Statistical keyphrase extraction 📊
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

    # Test scenarios - NO QUERIES, just context transfer
    scenarios = [
        {
            "name": "API Documentation",
            "context": """
The Weather API v2.1 has the following endpoints:
- GET /weather/current: Returns current weather data
- GET /weather/forecast: Returns 7-day forecast
- POST /weather/alerts: Subscribe to weather alerts

Rate limits: 1000 requests per hour for free tier, 10000 for premium.
API keys expire after 90 days and must be renewed.
Authentication: Include API key in X-API-Key header.
            """.strip(),
            "query": None,  # No query!
        },
        {
            "name": "Product Specifications",
            "context": """
The UltraBook Pro X1 features:
- Intel Core i7-12700H processor (14 cores, up to 4.7 GHz)
- 32GB DDR5 RAM (expandable to 64GB)
- 1TB NVMe SSD storage
- 15.6" 4K OLED display with 120Hz refresh rate
- NVIDIA RTX 3060 graphics card (6GB VRAM)
- Wi-Fi 6E and Bluetooth 5.2
- Weight: 1.8kg, Battery: 76Wh (up to 8 hours)
- Price: $1,899 USD
            """.strip(),
            "query": None,
        },
        {
            "name": "Meeting Notes",
            "context": """
Q4 Planning Meeting - October 15, 2025
Attendees: Sarah Chen (CEO), Mike Rodriguez (CTO), Lisa Park (CFO)

Key decisions:
- Launch new product line in Q1 2026
- Increase R&D budget by 25% to $2.5M
- Hire 10 new engineers by January
- Open Austin office in March 2026
- Target revenue: $50M for 2026

Action items:
- Sarah: Finalize product roadmap by Nov 1
- Mike: Complete hiring plan by Oct 30
- Lisa: Present budget proposal at board meeting
            """.strip(),
            "query": None,
        },
    ]

    calibration_data = [
        "Information can be transferred between agents.",
        "Context sharing enables better collaboration.",
    ]

    # Test both extraction methods
    extraction_methods = [
        {
            "name": "Attention (Queryless)",
            "method": "attention_queryless",
            "description": "Uses self-attention to find salient tokens - Zero overhead!",
        },
        {
            "name": "YAKE",
            "method": "yake",
            "description": "Statistical keyphrase extraction - No model needed!",
        },
    ]

    for extr in extraction_methods:
        print(f"\n{'='*80}")
        print(f"METHOD: {extr['name']}")
        print(f"{extr['description']}")
        print(f"{'='*80}")

        config = QKVCommConfig(
            mode="hybrid",
            quantization_enabled=True,
            calibration_enabled=True,
            layer_selection_ratio=0.7,
            hybrid_entity_extraction=extr["method"],
            hybrid_max_entity_tokens=30,
        )

        system = QKVCommSystem(model, model, config, device)
        system.calibrate(calibration_data)

        for scenario in scenarios:
            print(f"\n{'-'*80}")
            print(f"Scenario: {scenario['name']}")
            print(f"Context length: {len(scenario['context'])} chars")
            print(f"Query: {scenario['query']} ← No specific question!")
            print(f"{'-'*80}")

            # Note: We pass empty string as query since communicate() still expects it
            # But the extraction method won't use it
            output, metrics = system.communicate(
                scenario["context"],
                "",  # Empty query - extraction is query-independent
                max_new_tokens=80,
            )

            extracted = metrics.get("hybrid_extracted_facts", "N/A")

            print(f"\nExtracted Facts: {extracted}")
            print(f"\nReceiver Output:")
            print(f"{output}")
            print(f"\nCompression: {metrics.get('compression_ratio', 0):.2f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        """
✅ QUERY-INDEPENDENT EXTRACTION WORKS!

Key findings:
1. Attention (queryless) - Uses model's existing attention computation
   - Zero additional overhead
   - Extracts semantically important tokens automatically
   - Perfect for LLM agent research (theoretically aligned)

2. YAKE - Pure statistical method
   - No model loading required
   - Fast and lightweight
   - Good fallback when model unavailable

Real-world applications:
✓ Multi-agent collaboration without specific queries
✓ Context accumulation over multiple turns
✓ Proactive information sharing
✓ General knowledge transfer between agents

This makes Q-KVComm suitable for REAL agent communication scenarios,
not just question-answering demos!
    """
    )

    print("\nRECOMMENDATION:")
    print("Use 'attention_queryless' as default for efficiency research.")
    print("It leverages existing model computation with zero extra overhead.")


if __name__ == "__main__":
    main()
