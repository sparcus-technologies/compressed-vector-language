"""
Compare Q-KVComm performance with traditional natural language communication
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_kvcomm import QKVCommConfig, QKVCommSystem

def setup_models(model_name="gpt2", device="cuda"):
    """Setup sender and receiver models"""
    print(f"Loading models: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    sender = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sender.tokenizer = tokenizer
    
    receiver = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    receiver.tokenizer = tokenizer
    
    return sender, receiver

def traditional_communication(sender, receiver, context, query, max_tokens=50):
    """Traditional natural language communication"""
    # Sender generates summary/response
    sender.eval()
    
    with torch.no_grad():
        # Sender processes context and generates
        inputs = sender.tokenizer(
            f"Context: {context}\nSummary:",
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(sender.device)
        
        sender_output = sender.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=sender.tokenizer.eos_token_id
        )
        
        sender_text = sender.tokenizer.decode(
            sender_output[0],
            skip_special_tokens=True
        )
    
    # Receiver uses sender's output + query
    receiver.eval()
    
    with torch.no_grad():
        combined_input = f"{sender_text}\nQuery: {query}\nAnswer:"
        inputs = receiver.tokenizer(
            combined_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(receiver.device)
        
        receiver_output = receiver.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=receiver.tokenizer.eos_token_id
        )
        
        final_text = receiver.tokenizer.decode(
            receiver_output[0],
            skip_special_tokens=True
        )
    
    # Compute communication cost (tokens transmitted)
    sender_tokens = sender_output.shape[1]
    
    return final_text, sender_tokens

def run_comparison():
    """Run comprehensive comparison"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup models (using GPT-2 for demonstration)
    sender, receiver = setup_models("gpt2", device)
    
    # Calibration data
    calibration_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Climate change poses significant challenges to our planet.",
        "Quantum computing harnesses quantum mechanics for computation.",
        "The human brain contains approximately 86 billion neurons."
    ]
    
    # Test scenarios
    test_scenarios = [
        {
            "context": "Artificial intelligence has made remarkable progress in recent years. "
                      "Deep learning models can now perform complex tasks like image recognition, "
                      "natural language processing, and game playing at superhuman levels.",
            "query": "What are the main applications of AI mentioned?"
        },
        {
            "context": "Climate change is caused by greenhouse gas emissions. "
                      "The primary sources include burning fossil fuels, deforestation, "
                      "and industrial processes. Rising temperatures lead to extreme weather.",
            "query": "What causes climate change?"
        },
        {
            "context": "The solar system consists of the Sun and eight planets. "
                      "Mercury is closest to the Sun, while Neptune is the farthest. "
                      "Earth is the only planet known to support life.",
            "query": "Which planet is closest to the Sun?"
        }
    ]
    
    # Results storage
    results = []
    
    # Test different Q-KVComm configurations
    configs = [
        ("Baseline (No Compression)", QKVCommConfig(mode="baseline")),
        ("Quantization Only", QKVCommConfig(mode="quantization_only", target_bits=6.0)),
        ("Full Q-KVComm", QKVCommConfig(mode="full", target_bits=6.0)),
        ("Aggressive Compression", QKVCommConfig(mode="full", target_bits=4.5, 
                                                 min_bits=4, max_bits=6))
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON TESTS")
    print("=" * 80)
    
    for config_name, config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config_name}")
        print(f"{'=' * 80}")
        
        # Setup Q-KVComm system
        qkvcomm = QKVCommSystem(sender, receiver, config, device)
        qkvcomm.calibrate(calibration_data)
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\nScenario {i+1}:")
            print(f"Context: {scenario['context'][:100]}...")
            print(f"Query: {scenario['query']}")
            
            # Q-KVComm communication
            start_time = time.time()
            output, metrics = qkvcomm.communicate(
                scenario['context'],
                scenario['query'],
                max_new_tokens=30
            )
            qkvcomm_time = time.time() - start_time
            
            print(f"\nQ-KVComm Output: {output[:150]}...")
            print(f"Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
            print(f"Layers Transmitted: {metrics['num_layers_transmitted']}")
            print(f"Time: {qkvcomm_time:.3f}s")
            
            results.append({
                'Method': config_name,
                'Scenario': i + 1,
                'Compression Ratio': metrics['avg_compression_ratio'],
                'Layers Transmitted': metrics['num_layers_transmitted'],
                'Time (s)': qkvcomm_time,
                'Communication Cost': metrics['total_bits_compressed'] / 8 / 1024,  # KB
            })
    
    # Traditional communication comparison
    print(f"\n{'=' * 80}")
    print("Testing: Traditional Natural Language Communication")
    print(f"{'=' * 80}")
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}:")
        
        start_time = time.time()
        output, tokens = traditional_communication(
            sender, receiver,
            scenario['context'],
            scenario['query'],
            max_tokens=30
        )
        trad_time = time.time() - start_time
        
        print(f"Traditional Output: {output[:150]}...")
        print(f"Tokens Transmitted: {tokens}")
        print(f"Time: {trad_time:.3f}s")
        
        # Estimate communication cost (tokens * 2 bytes per token roughly)
        comm_cost = tokens * 2 / 1024  # KB
        
        results.append({
            'Method': 'Traditional NL',
            'Scenario': i + 1,
            'Compression Ratio': 1.0,
            'Layers Transmitted': 0,
            'Time (s)': trad_time,
            'Communication Cost': comm_cost,
        })
    
    # Create visualizations
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Q-KVComm vs Traditional Communication Performance', fontsize=16)
    
    # Compression ratio by method
    ax1 = axes[0, 0]
    compression_data = df.groupby('Method')['Compression Ratio'].mean().reset_index()
    sns.barplot(data=compression_data, x='Method', y='Compression Ratio', ax=ax1)
    ax1.set_title('Average Compression Ratio by Method')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    
    # Communication cost
    ax2 = axes[0, 1]
    cost_data = df.groupby('Method')['Communication Cost'].mean().reset_index()
    sns.barplot(data=cost_data, x='Method', y='Communication Cost', ax=ax2)
    ax2.set_title('Average Communication Cost (KB)')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=45)
    
    # Time comparison
    ax3 = axes[1, 0]
    time_data = df.groupby('Method')['Time (s)'].mean().reset_index()
    sns.barplot(data=time_data, x='Method', y='Time (s)', ax=ax3)
    ax3.set_title('Average Processing Time')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    
    # Compression vs Cost tradeoff
    ax4 = axes[1, 1]
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        ax4.scatter(
            method_data['Compression Ratio'],
            method_data['Communication Cost'],
            label=method,
            s=100,
            alpha=0.6
        )
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('Communication Cost (KB)')
    ax4.set_title('Compression-Cost Tradeoff')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qkvcomm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: qkvcomm_comparison.png")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("\nAverage metrics by method:")
    summary = df.groupby('Method').agg({
        'Compression Ratio': 'mean',
        'Communication Cost': 'mean',
        'Time (s)': 'mean'
    }).round(3)
    print(summary)
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    qkvcomm_full = df[df['Method'] == 'Full Q-KVComm']
    traditional = df[df['Method'] == 'Traditional NL']
    
    if len(qkvcomm_full) > 0 and len(traditional) > 0:
        cost_reduction = (1 - qkvcomm_full['Communication Cost'].mean() / 
                         traditional['Communication Cost'].mean()) * 100
        
        print(f"\n✓ Full Q-KVComm reduces communication cost by {cost_reduction:.1f}% "
              f"vs traditional NL")
        print(f"✓ Average compression ratio: "
              f"{qkvcomm_full['Compression Ratio'].mean():.2f}x")
        print(f"✓ Aggressive compression achieves up to "
              f"{df[df['Method']=='Aggressive Compression']['Compression Ratio'].mean():.2f}x")

if __name__ == "__main__":
    run_comparison()