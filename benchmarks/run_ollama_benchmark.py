"""Run benchmark with local models - 5 datasets with proper agentic metrics"""

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
    print("=" * 80)
    print("\n📚 Datasets:")
    print("  1. SQuAD 2.0 - Extractive QA (answerable + unanswerable)")
    print("  2. HotpotQA - Multi-hop reasoning")
    print("  3. Natural Questions - Open domain QA")
    print("  4. CoQA - Conversational QA")
    print("  5. NarrativeQA - Reading comprehension")
    print("\n📊 Evaluation Framework:")
    print("  • Contextual Relevance (question-answer relevance)")
    print("  • Answer Completeness (coverage of ground truth)")
    print("  • Semantic Fidelity (meaning preservation)")
    print("  • Response Coherence (output quality)")
    print("  • Communication Efficiency (quality per bit)")
    print("  • Information Throughput (quality per second)")
    print("\n🔬 Compression Analysis:")
    print("  • Compression Quality Score (vs baseline)")
    print("  • Semantic Preservation (compressed vs uncompressed)")
    print("  • Bandwidth savings and layer efficiency")
    print("=" * 80)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    # Model selection - use instruction-tuned for better agentic communication
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"🤖 Model: {model_name}")

    # Load tokenizer
    print("\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Load sender model
    print("\n⏳ Loading sender model...")
    sender = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
    )
    sender.tokenizer = tokenizer
    print("✓ Sender model loaded")

    # Load receiver model (same for homogeneous setup)
    print("\n⏳ Loading receiver model...")
    receiver = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map=device,
    )
    receiver.tokenizer = tokenizer
    print("✓ Receiver model loaded")

    # Configure Q-KVComm for optimal agentic communication
    print("\n⚙️  Configuring Q-KVComm system...")
    config = QKVCommConfig(
        # Mode
        mode="full",  # Full Q-KVComm with quantization + calibration
        
        # Quantization settings - balanced for quality/compression
        target_bits=6.0,  # Target 6-bit (can adjust: 4-8 range)
        min_bits=4,
        max_bits=8,
        quantization_enabled=True,
        profiling_samples=50,  # Samples for sensitivity profiling
        
        # Calibration settings
        calibration_enabled=True,
        calibration_samples=30,  # Samples for feature calibration
        
        # Layer selection - preserve more layers for quality
        layer_selection_ratio=0.7,  # Use 70% of layers
        attention_weight=0.5,  # Balance attention + Gaussian prior
        
        # Information extraction
        extraction_method="yake",  # YAKE for keyword extraction
        extraction_max_tokens=50,
        extraction_min_confidence=0.5,
        extraction_cache_enabled=True,
        
        # Memory management
        max_memory_mb=1024.0,  # 1GB cache
        enable_disk_cache=True,
        adaptive_compression=True,
    )
    
    print("✓ Configuration loaded")
    print(f"  • Target bits: {config.target_bits}")
    print(f"  • Layer selection: {config.layer_selection_ratio * 100:.0f}%")
    print(f"  • Extraction method: {config.extraction_method}")

    # Initialize Q-KVComm system
    print("\n🔧 Initializing Q-KVComm system...")
    qkvcomm = QKVCommSystem(sender, receiver, config, device)
    print("✓ Q-KVComm system initialized")

    # Create benchmark suite
    print("\n📋 Setting up benchmark suite...")
    
    # Set enable_baseline=True to compare compressed vs uncompressed
    # This is CRITICAL for research evaluation but slower (2x time)
    enable_baseline = True
    
    benchmark = BenchmarkSuite(
        qkvcomm_system=qkvcomm,
        output_dir="benchmark_results",
        enable_baseline=enable_baseline,
    )
    print("✓ Benchmark suite ready")
    
    if enable_baseline:
        print("\n⚠️  BASELINE COMPARISON ENABLED")
        print("  • Will compare compressed vs uncompressed outputs")
        print("  • Provides compression quality metrics")
        print("  • Evaluation time: ~2x longer")
        print("  • Recommended for research/publication")
    else:
        print("\n⚡ FAST MODE (no baseline)")
        print("  • Evaluates compressed output only")
        print("  • Faster evaluation")
        print("  • Good for quick testing")

    # Run benchmark
    print("\n" + "=" * 80)
    print("STARTING BENCHMARK EVALUATION")
    print("=" * 80)
    
    benchmark.run_benchmark(
        dataset_names=[
            "squad",           # Extractive QA
            "hotpot_qa",       # Multi-hop reasoning
            "natural_questions",  # Open domain QA
            "coqa",            # Conversational QA
            "narrativeqa",     # Reading comprehension
        ],
        max_samples=5,  # Samples per dataset (increase for full eval)
        max_new_tokens=50,  # Max tokens to generate
    )

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
    print("\n📁 Results saved to:")
    print("  • benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.json")
    print("  • benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.csv")
    print("\n📊 To visualize results, run:")
    print("  python visualize_benchmark.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()