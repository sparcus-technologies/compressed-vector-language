"""
Demonstration of proper information extraction in Q-KVComm
Shows YAKE, SpaCy, and Hybrid methods with real examples
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem
from q_kvcomm.adaptive_extraction import InformationExtractor, ContextTypeDetector


def demo_extraction_methods():
    """Compare different extraction methods"""
    
    print("=" * 80)
    print("INFORMATION EXTRACTION COMPARISON")
    print("=" * 80)
    
    # Test contexts
    contexts = {
        "API Documentation": """
The TurboMax API v2.1 has a rate limit of 500 requests per minute.
Authentication requires an API key in the X-Auth-Token header.
The base endpoint is https://api.turbomax.com/v2.
Supported methods: GET /users, POST /users, DELETE /users/{id}.
Error responses use standard HTTP status codes (400, 401, 500).
        """.strip(),
        
        "Product Specification": """
The UltraBook Pro X1 launched in Q3 2025 at $1,899 USD.
Features Intel Core i7-12700H (14 cores), 32GB DDR5 RAM,
1TB NVMe SSD, 15.6" 4K OLED display with 120Hz refresh rate.
NVIDIA RTX 3060 GPU with 6GB VRAM. Weight: 1.8kg.
Battery: 76Wh lasting approximately 8 hours.
Connectivity: Wi-Fi 6E, Bluetooth 5.2, Thunderbolt 4 (x2).
        """.strip(),
        
        "Technical Specification": """
The system requires Python 3.8+ with PyTorch 2.0.0 or higher.
Minimum 16GB RAM recommended, 32GB for optimal performance.
GPU support: CUDA 11.7+ with compute capability 7.0+.
Storage: 50GB free space for models and cache.
Network: 100 Mbps for model downloads.
        """.strip()
    }
    
    methods = ['yake', 'spacy_ner', 'hybrid', 'simple']
    
    for context_name, context in contexts.items():
        print(f"\n{'='*80}")
        print(f"CONTEXT: {context_name}")
        print(f"{'='*80}")
        print(f"Text: {context[:100]}...")
        
        # Detect context type
        detector = ContextTypeDetector()
        detected_type = detector.detect(context)
        print(f"\nDetected Type: {detected_type}")
        
        for method in methods:
            print(f"\n{'-'*80}")
            print(f"Method: {method.upper()}")
            print(f"{'-'*80}")
            
            try:
                extractor = InformationExtractor(extraction_method=method)
                facts = extractor.extract_facts(context)
                
                # Format for display
                formatted = extractor.format_facts_for_transmission(
                    facts,
                    max_tokens=50,
                    min_confidence=0.5
                )
                
                print(f"Extracted {len(facts)} facts")
                print(f"Formatted ({len(formatted)} chars): {formatted}")
                
                # Show top 5 facts with details
                print("\nTop 5 Facts:")
                for i, fact in enumerate(facts[:5], 1):
                    print(f"  {i}. [{fact.fact_type}] {fact.content} "
                          f"(confidence: {fact.confidence:.2f})")
                
            except Exception as e:
                print(f"ERROR: {e}")
        
        print()


def demo_end_to_end():
    """End-to-end demo with proper extraction"""
    
    print("\n" + "=" * 80)
    print("END-TO-END Q-KVCOMM WITH PROPER EXTRACTION")
    print("=" * 80)
    
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
    
    # Test scenarios
    scenarios = [
        {
            "name": "API Query with YAKE",
            "context": """
The DataStream API v3.0 supports real-time data processing.
Rate limit: 1000 requests/minute for premium accounts.
Authentication: Bearer token in Authorization header.
Base URL: https://api.datastream.io/v3
Endpoints: GET /stream, POST /process, DELETE /session/{id}
            """.strip(),
            "query": "What's the rate limit?",
            "method": "yake"
        },
        {
            "name": "Product Query with Hybrid",
            "context": """
The MegaPhone Z9 released in September 2025 for $999.
Specs: Snapdragon 8 Gen 3, 12GB RAM, 256GB storage.
Display: 6.7" AMOLED at 144Hz with 2K resolution.
Camera: 200MP main, 50MP ultra-wide, 12MP telephoto.
Battery: 5000mAh with 65W fast charging.
            """.strip(),
            "query": "How much does it cost?",
            "method": "hybrid"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*80}")
        print(f"Context: {scenario['context'][:100]}...")
        print(f"Query: {scenario['query']}")
        print(f"Method: {scenario['method']}")
        
        # Configure with proper extraction
        config = QKVCommConfig(
            mode="hybrid",
            quantization_enabled=True,
            calibration_enabled=True,
            extraction_method=scenario['method'],
            extraction_max_tokens=30,
            extraction_cache_enabled=True,
            max_memory_mb=512
        )
        
        # Initialize system
        system = QKVCommSystem(model, model, config, device)
        
        # Calibration
        calibration_data = [scenario['context'], "Sample calibration text"]
        system.calibrate(calibration_data)
        
        # Communicate
        output, metrics = system.communicate(
            scenario['context'],
            scenario['query'],
            max_new_tokens=50
        )
        
        print(f"\n{'-'*80}")
        print("RESULTS")
        print(f"{'-'*80}")
        print(f"Extracted Facts: {metrics.get('extracted_facts', 'N/A')}")
        print(f"Num Facts: {metrics.get('num_facts_extracted', 0)}")
        print(f"Cache Hit: {metrics.get('extraction_cache_hit', False)}")
        print(f"Compression: {metrics.get('overall_compression_ratio', 1.0):.2f}x")
        print(f"\nGenerated: {output}")
        
        # Memory stats
        mem_stats = metrics.get('memory_stats', {})
        if mem_stats:
            print(f"\nMemory Stats:")
            print(f"  Hit Rate: {mem_stats.get('hit_rate', 0):.1%}")
            print(f"  Memory: {mem_stats.get('current_memory_mb', 0):.1f} MB")
            print(f"  Entries: {mem_stats.get('num_entries', 0)}")


if __name__ == "__main__":
    print("\n🔥 PROPER INFORMATION EXTRACTION FOR Q-KVCOMM\n")
    
    # Demo 1: Compare extraction methods
    demo_extraction_methods()
    
    # Demo 2: End-to-end with extraction
    demo_end_to_end()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
✅ PRODUCTION USE:
   - Use 'hybrid' method for best accuracy
   - Use 'yake' method for best speed/accuracy tradeoff
   - Enable extraction caching for repeated contexts
   - Set max_memory_mb based on your system

📊 RESEARCH USE:
   - Use 'yake' as baseline (zero model overhead)
   - Compare against 'simple' for ablation studies
   - Track extraction cache hit rates
   - Measure extraction time vs. accuracy

🚀 OPTIMIZATION:
   - Tune extraction_max_tokens based on your task
   - Adjust extraction_min_confidence for precision/recall
   - Enable adaptive_compression for memory pressure
   - Use disk caching for large deployments
    """)