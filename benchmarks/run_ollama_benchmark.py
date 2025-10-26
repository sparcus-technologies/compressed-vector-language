"""Run benchmark with Ollama local models"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from q_kvcomm import QKVCommConfig, QKVCommSystem
from benchmark_suite import BenchmarkSuite


def main():
    print("Loading Ollama-compatible models...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use TinyLlama - smallest good instruction-tuned model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading sender model...")
    sender = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    sender = sender.to(device)
    sender.tokenizer = tokenizer
    
    print("Loading receiver model...")
    receiver = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    receiver = receiver.to(device)
    receiver.tokenizer = tokenizer
    
    # Configure Q-KVComm
    config = QKVCommConfig(
        mode='full',
        target_bits=6.0,
        layer_selection_ratio=0.6,
        quantization_enabled=True,
        calibration_enabled=True,
        profiling_samples=30,  # Reduced for faster testing
        calibration_samples=20
    )
    
    qkvcomm = QKVCommSystem(sender, receiver, config, device)
    
    # Create benchmark - ONLY qkvcomm_system and output_dir parameters
    benchmark = BenchmarkSuite(
        qkvcomm_system=qkvcomm,
        output_dir='benchmark_results'
    )
    
    print(f"\nRunning benchmark with {model_name}...")
    
    # Run benchmark with correct method signature
    benchmark.run_benchmark(
        dataset_names=['squad'],
        max_samples=20,  # Small number for testing
        max_new_tokens=30
    )
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()