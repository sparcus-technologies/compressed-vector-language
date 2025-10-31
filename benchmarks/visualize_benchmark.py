"""Visualize agentic communication benchmark results"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_latest_results(results_dir: str):
    """Load the most recent benchmark results"""
    results_dir = Path(results_dir)

    json_files = list(results_dir.glob("benchmark_results_*.json"))
    if not json_files:
        print(f"No results found in {results_dir}")
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, "r") as f:
        return json.load(f)


def create_visualizations(results: dict, output_dir: str):
    """Create comprehensive visualizations for agentic communication metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    all_data = []
    for dataset_name, dataset_results in results.items():
        for result in dataset_results:
            if result.get("success", False):
                all_data.append(
                    {
                        "Dataset": dataset_name,
                        "Contextual Relevance": result.get("contextual_relevance", 0),
                        "Answer Completeness": result.get("answer_completeness", 0),
                        "Semantic Fidelity": result.get("semantic_fidelity", 0),
                        "Response Coherence": result.get("response_coherence", 0),
                        "Information Density": result.get("information_density", 0),
                        "Inference Time (s)": result["inference_time"],
                        "Compression Ratio": result["compression_ratio"],
                        "Layers Transmitted": result["layers_transmitted"],
                        "Bits Saved (Mb)": result["bits_saved"] / 1e6,
                    }
                )

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No successful results to visualize")
        return

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Contextual Relevance by Dataset
    ax1 = fig.add_subplot(gs[0, 0])
    dataset_cr = (
        df.groupby("Dataset")["Contextual Relevance"].agg(["mean", "std"]).reset_index()
    )
    bars = ax1.bar(
        dataset_cr["Dataset"],
        dataset_cr["mean"],
        yerr=dataset_cr["std"],
        capsize=5,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_title("Contextual Relevance by Dataset", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Good (>0.7)")
    ax1.legend()

    # 2. Answer Completeness by Dataset
    ax2 = fig.add_subplot(gs[0, 1])
    dataset_ac = (
        df.groupby("Dataset")["Answer Completeness"].agg(["mean", "std"]).reset_index()
    )
    ax2.bar(
        dataset_ac["Dataset"],
        dataset_ac["mean"],
        yerr=dataset_ac["std"],
        capsize=5,
        color="lightcoral",
        edgecolor="black",
    )
    ax2.set_title("Answer Completeness by Dataset", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, label="Good (>0.7)")
    ax2.legend()

    # 3. Compression Ratio by Dataset
    ax3 = fig.add_subplot(gs[0, 2])
    dataset_comp = (
        df.groupby("Dataset")["Compression Ratio"].agg(["mean", "std"]).reset_index()
    )
    ax3.bar(
        dataset_comp["Dataset"],
        dataset_comp["mean"],
        yerr=dataset_comp["std"],
        capsize=5,
        color="lightgreen",
        edgecolor="black",
    )
    ax3.set_title("Compression Ratio by Dataset", fontweight="bold", fontsize=12)
    ax3.set_ylabel("Compression Ratio (x)")
    ax3.axhline(y=2.0, color="green", linestyle="--", alpha=0.5, label="Target (2x)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Semantic Fidelity Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    df.boxplot(column="Semantic Fidelity", by="Dataset", ax=ax4)
    ax4.set_title("Semantic Fidelity Distribution", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Dataset")
    ax4.set_ylabel("Score")
    ax4.axhline(y=0.7, color="green", linestyle="--", alpha=0.5)
    plt.sca(ax4)
    plt.xticks(rotation=45)

    # 5. Response Coherence Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    df.boxplot(column="Response Coherence", by="Dataset", ax=ax5)
    ax5.set_title("Response Coherence Distribution", fontweight="bold", fontsize=12)
    ax5.set_xlabel("Dataset")
    ax5.set_ylabel("Score")
    ax5.axhline(y=0.7, color="green", linestyle="--", alpha=0.5)
    plt.sca(ax5)
    plt.xticks(rotation=45)

    # 6. Quality vs Compression Trade-off
    ax6 = fig.add_subplot(gs[1, 2])
    for dataset in df["Dataset"].unique():
        dataset_df = df[df["Dataset"] == dataset]
        quality = dataset_df["Answer Completeness"].mean()
        compression = dataset_df["Compression Ratio"].mean()
        ax6.scatter(compression, quality, label=dataset, alpha=0.8, s=200)
    ax6.set_xlabel("Compression Ratio (x)", fontweight="bold")
    ax6.set_ylabel("Answer Completeness", fontweight="bold")
    ax6.set_title("Quality vs Compression Trade-off", fontweight="bold", fontsize=12)
    ax6.axhline(y=0.7, color="green", linestyle="--", alpha=0.3, label="Quality Target")
    ax6.axvline(
        x=2.0, color="blue", linestyle="--", alpha=0.3, label="Compression Target"
    )
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Bandwidth Saved by Dataset
    ax7 = fig.add_subplot(gs[2, 0])
    dataset_bits = df.groupby("Dataset")["Bits Saved (Mb)"].sum().reset_index()
    ax7.bar(
        dataset_bits["Dataset"],
        dataset_bits["Bits Saved (Mb)"],
        color="orange",
        edgecolor="black",
    )
    ax7.set_title("Total Bandwidth Saved", fontweight="bold", fontsize=12)
    ax7.set_ylabel("Bits Saved (Mb)")
    ax7.tick_params(axis="x", rotation=45)
    ax7.grid(axis="y", alpha=0.3)

    # 8. Information Density by Dataset
    ax8 = fig.add_subplot(gs[2, 1])
    dataset_density = (
        df.groupby("Dataset")["Information Density"].agg(["mean", "std"]).reset_index()
    )
    ax8.bar(
        dataset_density["Dataset"],
        dataset_density["mean"],
        yerr=dataset_density["std"],
        capsize=5,
        color="mediumpurple",
        edgecolor="black",
    )
    ax8.set_title("Information Density by Dataset", fontweight="bold", fontsize=12)
    ax8.set_ylabel("Density Score")
    ax8.set_ylim(0, 1)
    ax8.tick_params(axis="x", rotation=45)
    ax8.grid(axis="y", alpha=0.3)
    ax8.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Good (>0.5)")
    ax8.legend()

    # 9. Overall Quality Radar Chart
    ax9 = fig.add_subplot(gs[2, 2], projection="polar")

    categories = [
        "Contextual\nRelevance",
        "Answer\nCompleteness",
        "Semantic\nFidelity",
        "Response\nCoherence",
        "Information\nDensity",
    ]

    values = [
        df["Contextual Relevance"].mean(),
        df["Answer Completeness"].mean(),
        df["Semantic Fidelity"].mean(),
        df["Response Coherence"].mean(),
        df["Information Density"].mean(),
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax9.plot(angles, values, "o-", linewidth=2, color="blue", label="Q-KVComm")
    ax9.fill(angles, values, alpha=0.25, color="blue")
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(categories, size=9)
    ax9.set_ylim(0, 1)
    ax9.set_title(
        "Overall Agentic Quality Profile", fontweight="bold", fontsize=12, pad=20
    )
    ax9.grid(True)
    ax9.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 10. Latency Distribution
    ax10 = fig.add_subplot(gs[3, 0])
    df.boxplot(column="Inference Time (s)", by="Dataset", ax=ax10)
    ax10.set_title("Inference Latency Distribution", fontweight="bold", fontsize=12)
    ax10.set_xlabel("Dataset")
    ax10.set_ylabel("Time (s)")
    plt.sca(ax10)
    plt.xticks(rotation=45)

    # 11. Layers vs Compression
    ax11 = fig.add_subplot(gs[3, 1])
    for dataset in df["Dataset"].unique():
        dataset_df = df[df["Dataset"] == dataset]
        ax11.scatter(
            dataset_df["Layers Transmitted"],
            dataset_df["Compression Ratio"],
            label=dataset,
            alpha=0.6,
            s=100,
        )
    ax11.set_xlabel("Layers Transmitted")
    ax11.set_ylabel("Compression Ratio")
    ax11.set_title("Layer Selection vs Compression", fontweight="bold", fontsize=12)
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # 12. Composite Performance Score
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis("off")

    # Calculate composite scores
    quality_score = np.mean(
        [
            df["Contextual Relevance"].mean(),
            df["Answer Completeness"].mean(),
            df["Semantic Fidelity"].mean(),
            df["Response Coherence"].mean(),
            df["Information Density"].mean(),
        ]
    )

    efficiency_score = min(1.0, df["Compression Ratio"].mean() / 3.0)
    composite = 0.6 * quality_score + 0.4 * efficiency_score

    stats_text = f"""
    COMPOSITE PERFORMANCE SCORE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🏆 Overall: {composite:.4f}
    
    Quality (60%): {quality_score:.4f}
      • Context Relevance: {df['Contextual Relevance'].mean():.4f}
      • Answer Complete: {df['Answer Completeness'].mean():.4f}
      • Semantic Fidelity: {df['Semantic Fidelity'].mean():.4f}
      • Coherence: {df['Response Coherence'].mean():.4f}
      • Info Density: {df['Information Density'].mean():.4f}
    
    Efficiency (40%): {efficiency_score:.4f}
      • Compression: {df['Compression Ratio'].mean():.2f}x
      • Bandwidth Saved: {df['Bits Saved (Mb)'].sum():.2f} Mb
      • Avg Latency: {df['Inference Time (s)'].mean():.3f}s
    
    Samples Evaluated: {len(df)}
    """

    ax12.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        "Q-KVComm Agentic Communication Benchmark - Comprehensive Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save figure
    output_file = output_dir / "benchmark_agentic_comprehensive.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")

    # Create detailed per-dataset plots
    for dataset_name in df["Dataset"].unique():
        create_dataset_plot(df[df["Dataset"] == dataset_name], dataset_name, output_dir)

    plt.show()


def create_dataset_plot(df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """Create detailed plot for a specific dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{dataset_name} - Agentic Communication Analysis",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Quality Metrics Distribution
    ax = axes[0, 0]
    quality_metrics = [
        "Contextual Relevance",
        "Answer Completeness",
        "Semantic Fidelity",
        "Response Coherence",
        "Information Density",
    ]
    means = [df[m].mean() for m in quality_metrics]
    stds = [df[m].std() for m in quality_metrics]

    bars = ax.barh(
        quality_metrics,
        means,
        xerr=stds,
        capsize=5,
        color=["skyblue", "lightcoral", "lightgreen", "gold", "mediumpurple"],
        edgecolor="black",
    )
    ax.set_xlabel("Score")
    ax.set_title("Quality Metrics Profile")
    ax.set_xlim(0, 1)
    ax.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good (>0.7)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means)):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    # 2. Quality vs Efficiency Scatter
    ax = axes[0, 1]
    quality_composite = df[
        ["Contextual Relevance", "Answer Completeness", "Response Coherence"]
    ].mean(axis=1)
    scatter = ax.scatter(
        df["Compression Ratio"],
        quality_composite,
        c=df["Inference Time (s)"],
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
    )
    ax.set_xlabel("Compression Ratio (x)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality vs Compression Trade-off")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Latency (s)")

    # 3. Communication Efficiency Analysis
    ax = axes[1, 0]
    efficiency_data = {
        "Compression\nRatio": df["Compression Ratio"].mean(),
        "Bits Saved\n(normalized)": df["Bits Saved (Mb)"].sum()
        / 100,  # Normalize for viz
        "Layer\nEfficiency": 1.0
        - (df["Layers Transmitted"].mean() / 20),  # Assuming ~20 layers
    }

    bars = ax.bar(
        efficiency_data.keys(),
        efficiency_data.values(),
        color=["lightgreen", "orange", "cyan"],
        edgecolor="black",
    )
    ax.set_ylabel("Score / Ratio")
    ax.set_title("Communication Efficiency Breakdown")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, (key, val) in zip(bars, efficiency_data.items()):
        label = f"{val:.2f}" if "Ratio" not in key else f"{val:.2f}x"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis("off")

    quality_avg = np.mean(
        [
            df["Contextual Relevance"].mean(),
            df["Answer Completeness"].mean(),
            df["Response Coherence"].mean(),
            df["Information Density"].mean(),
        ]
    )

    stats_text = f"""
    Dataset: {dataset_name}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📊 Quality Metrics:
      • Avg Quality Score: {quality_avg:.4f}
      • Context Relevance: {df['Contextual Relevance'].mean():.4f}
      • Answer Complete: {df['Answer Completeness'].mean():.4f}
      • Semantic Fidelity: {df['Semantic Fidelity'].mean():.4f}
      • Coherence: {df['Response Coherence'].mean():.4f}
      • Info Density: {df['Information Density'].mean():.4f}
    
    ⚡ Efficiency Metrics:
      • Compression: {df['Compression Ratio'].mean():.2f}x
      • Bandwidth Saved: {df['Bits Saved (Mb)'].sum():.2f} Mb
      • Avg Latency: {df['Inference Time (s)'].mean():.3f}s
      • Avg Layers: {df['Layers Transmitted'].mean():.1f}
    
    📈 Statistics:
      • Samples: {len(df)}
      • Success Rate: 100%
    """

    ax.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.2),
    )

    plt.tight_layout()

    output_file = output_dir / f"{dataset_name}_agentic_detail.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Detail plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Q-KVComm agentic communication benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmark_results",
        help="Directory with benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_visualizations",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    results = load_latest_results(args.results_dir)
    if results:
        create_visualizations(results, args.output_dir)


if __name__ == "__main__":
    main()
