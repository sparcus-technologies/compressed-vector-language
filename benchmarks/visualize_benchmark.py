"""Visualize benchmark results"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse
from pathlib import Path


def load_latest_results(results_dir: str):
    """Load the most recent benchmark results"""
    results_dir = Path(results_dir)
    
    # Find latest JSON file
    json_files = list(results_dir.glob("benchmark_results_*.json"))
    if not json_files:
        print(f"No results found in {results_dir}")
        return None
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def create_visualizations(results: dict, output_dir: str):
    """Create comprehensive visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    all_data = []
    for dataset_name, dataset_results in results.items():
        for result in dataset_results:
            if result.get('success', False):
                all_data.append({
                    'Dataset': dataset_name,
                    'F1 Score': result['f1'],
                    'Exact Match': result['exact_match'],
                    'Inference Time (s)': result['inference_time'],
                    'Compression Ratio': result['compression_ratio'],
                    'Layers Transmitted': result['layers_transmitted'],
                    'Bits Saved (Mb)': result['bits_saved'] / 1e6
                })
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No successful results to visualize")
        return
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. F1 Score by Dataset
    ax1 = fig.add_subplot(gs[0, 0])
    dataset_f1 = df.groupby('Dataset')['F1 Score'].agg(['mean', 'std']).reset_index()
    ax1.bar(dataset_f1['Dataset'], dataset_f1['mean'], yerr=dataset_f1['std'], 
            capsize=5, color='skyblue', edgecolor='black')
    ax1.set_title('F1 Score by Dataset', fontweight='bold', fontsize=12)
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Exact Match by Dataset
    ax2 = fig.add_subplot(gs[0, 1])
    dataset_em = df.groupby('Dataset')['Exact Match'].agg(['mean', 'std']).reset_index()
    ax2.bar(dataset_em['Dataset'], dataset_em['mean'], yerr=dataset_em['std'],
            capsize=5, color='lightcoral', edgecolor='black')
    ax2.set_title('Exact Match by Dataset', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Exact Match')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Compression Ratio by Dataset
    ax3 = fig.add_subplot(gs[0, 2])
    dataset_comp = df.groupby('Dataset')['Compression Ratio'].agg(['mean', 'std']).reset_index()
    ax3.bar(dataset_comp['Dataset'], dataset_comp['mean'], yerr=dataset_comp['std'],
            capsize=5, color='lightgreen', edgecolor='black')
    ax3.set_title('Compression Ratio by Dataset', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Compression Ratio (x)')
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No Compression')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. F1 Score Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    df.boxplot(column='F1 Score', by='Dataset', ax=ax4)
    ax4.set_title('F1 Score Distribution', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('F1 Score')
    plt.sca(ax4)
    plt.xticks(rotation=45)
    
    # 5. Inference Time Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    df.boxplot(column='Inference Time (s)', by='Dataset', ax=ax5)
    ax5.set_title('Inference Time Distribution', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Dataset')
    ax5.set_ylabel('Time (s)')
    plt.sca(ax5)
    plt.xticks(rotation=45)
    
    # 6. Compression vs Quality Trade-off
    ax6 = fig.add_subplot(gs[1, 2])
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        ax6.scatter(dataset_df['Compression Ratio'], dataset_df['F1 Score'],
                   label=dataset, alpha=0.6, s=100)
    ax6.set_xlabel('Compression Ratio (x)', fontweight='bold')
    ax6.set_ylabel('F1 Score', fontweight='bold')
    ax6.set_title('Quality vs Compression Trade-off', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Bits Saved by Dataset
    ax7 = fig.add_subplot(gs[2, 0])
    dataset_bits = df.groupby('Dataset')['Bits Saved (Mb)'].sum().reset_index()
    ax7.bar(dataset_bits['Dataset'], dataset_bits['Bits Saved (Mb)'],
            color='orange', edgecolor='black')
    ax7.set_title('Total Communication Saved', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Bits Saved (Mb)')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Layers Transmitted Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    df.boxplot(column='Layers Transmitted', by='Dataset', ax=ax8)
    ax8.set_title('Layers Transmitted Distribution', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Dataset')
    ax8.set_ylabel('Number of Layers')
    plt.sca(ax8)
    plt.xticks(rotation=45)
    
    # 9. Overall Performance Summary
    ax9 = fig.add_subplot(gs[2, 2])
    metrics = ['F1 Score', 'Exact Match', 'Compression Ratio']
    values = [df['F1 Score'].mean(), df['Exact Match'].mean(), 
              df['Compression Ratio'].mean() / 10]  # Scale compression for visualization
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax9.barh(metrics, values, color=colors, edgecolor='black')
    ax9.set_xlabel('Score / Ratio', fontweight='bold')
    ax9.set_title('Overall Performance Summary', fontweight='bold', fontsize=12)
    ax9.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i == 2:  # Compression ratio
            label = f"{val*10:.2f}x"
        else:
            label = f"{val:.3f}"
        ax9.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                label, va='center')
    
    plt.suptitle('Q-KVComm Benchmark Results - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = output_dir / 'benchmark_comprehensive.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Create detailed per-dataset plots
    for dataset_name in df['Dataset'].unique():
        create_dataset_plot(df[df['Dataset'] == dataset_name], dataset_name, output_dir)
    
    plt.show()


def create_dataset_plot(df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """Create detailed plot for a specific dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} - Detailed Analysis', fontsize=14, fontweight='bold')
    
    # 1. Score Distribution
    ax = axes[0, 0]
    ax.hist([df['F1 Score'], df['Exact Match']], label=['F1 Score', 'Exact Match'],
            bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Quality Metrics Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Performance Scatter
    ax = axes[0, 1]
    scatter = ax.scatter(df['Inference Time (s)'], df['F1 Score'],
                        c=df['Compression Ratio'], cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Inference Time (s)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance Analysis')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Compression Ratio')
    
    # 3. Compression Analysis
    ax = axes[1, 0]
    ax.scatter(df['Layers Transmitted'], df['Compression Ratio'],
              alpha=0.6, s=100, color='coral', edgecolors='black')
    ax.set_xlabel('Layers Transmitted')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Strategy')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Dataset: {dataset_name}
    
    Quality Metrics:
      • Mean F1: {df['F1 Score'].mean():.4f} (±{df['F1 Score'].std():.4f})
      • Mean EM: {df['Exact Match'].mean():.4f} (±{df['Exact Match'].std():.4f})
    
    Performance:
      • Avg Time: {df['Inference Time (s)'].mean():.3f}s
      • Compression: {df['Compression Ratio'].mean():.2f}x
      • Avg Layers: {df['Layers Transmitted'].mean():.1f}
    
    Communication:
      • Total Saved: {df['Bits Saved (Mb)'].sum():.2f} Mb
      • Samples: {len(df)}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_file = output_dir / f'{dataset_name}_detail.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Detail plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Q-KVComm benchmark results')
    parser.add_argument('--results-dir', type=str, default='benchmark_results',
                       help='Directory with benchmark results')
    parser.add_argument('--output-dir', type=str, default='benchmark_visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results = load_latest_results(args.results_dir)
    if results:
        create_visualizations(results, args.output_dir)


if __name__ == "__main__":
    main()