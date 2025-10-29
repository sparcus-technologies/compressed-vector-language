"""Run benchmark with Ollama local models - 5 datasets"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from q_kvcomm import QKVCommConfig, QKVCommSystem
from benchmark_suite import BenchmarkSuite


def main():
    print("="*80)
    print("Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK")
    print("5 Datasets: SQuAD, HotpotQA, Natural Questions, CoQA, NarrativeQA")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use instruction-tuned model for better agentic communication
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading sender model...")
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
    
    # Configure Q-KVComm for optimal agentic communication
    config = QKVCommConfig(
        mode='full',
        target_bits=6.0,
        layer_selection_ratio=0.6,
        quantization_enabled=True,
        calibration_enabled=True,
        profiling_samples=30,
        calibration_samples=20
    )
    
    qkvcomm = QKVCommSystem(sender, receiver, config, device)
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(
        qkvcomm_system=qkvcomm,
        output_dir='benchmark_results'
    )
    
    print(f"\nRunning benchmark across 5 datasets...")
    
    # Run benchmark with 5 diverse datasets
    benchmark.run_benchmark(
        dataset_names=[
            'squad',              # Extractive QA
            'hotpot_qa',          # Multi-hop reasoning
            'natural_questions',  # Open domain QA
            'coqa',               # Conversational QA
            'narrativeqa'         # Long-form reading comprehension
        ],
        max_samples=20,  # 20 samples per dataset = 100 total
        max_new_tokens=50
    )
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE - Results saved to benchmark_results/")
    print("="*80)


if __name__ == "__main__":
    main()