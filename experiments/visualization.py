"""
Visualization script for publication-quality figures

Generates all plots needed for the research paper from experiment results.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"


class ExperimentVisualizer:
    """Create publication-quality visualizations"""

    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        print(f"📊 Q-KVComm Experiment Visualizer")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🖼️  Figures will be saved to: {self.figures_dir}\n")

    def plot_experiment_1_compression_quality(self):
        """Figure 1: Compression vs Quality Trade-off"""
        file_path = self.results_dir / "experiment_1_compression_quality.csv"
        if not file_path.exists():
            print(f"⊘ Skipping Experiment 1: {file_path} not found")
            return

        df = pd.read_csv(file_path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Compression vs Quality Trade-off", fontsize=16, fontweight="bold")

        # Plot 1: Contextual Relevance
        ax = axes[0, 0]
        for dataset in df["dataset"].unique():
            data = df[df["dataset"] == dataset]
            ax.plot(
                data["target_bits"],
                data["contextual_relevance"],
                marker="o",
                label=dataset,
                linewidth=2,
            )
        ax.set_xlabel("Target Bits")
        ax.set_ylabel("Contextual Relevance")
        ax.set_title("Contextual Relevance vs Compression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Answer Completeness
        ax = axes[0, 1]
        for dataset in df["dataset"].unique():
            data = df[df["dataset"] == dataset]
            ax.plot(
                data["target_bits"],
                data["answer_completeness"],
                marker="s",
                label=dataset,
                linewidth=2,
            )
        ax.set_xlabel("Target Bits")
        ax.set_ylabel("Answer Completeness")
        ax.set_title("Answer Completeness vs Compression")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Compression Ratio
        ax = axes[1, 0]
        for dataset in df["dataset"].unique():
            data = df[df["dataset"] == dataset]
            ax.plot(
                data["target_bits"],
                data["compression_ratio"],
                marker="^",
                label=dataset,
                linewidth=2,
            )
        ax.set_xlabel("Target Bits")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Achieved Compression Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Bandwidth Saved
        ax = axes[1, 1]
        bandwidth_by_bits = df.groupby("target_bits")["bandwidth_saved_mb"].sum()
        ax.bar(
            bandwidth_by_bits.index,
            bandwidth_by_bits.values,
            color="steelblue",
            alpha=0.7,
        )
        ax.set_xlabel("Target Bits")
        ax.set_ylabel("Total Bandwidth Saved (MB)")
        ax.set_title("Bandwidth Savings by Compression Level")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = self.figures_dir / "fig1_compression_quality.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_experiment_2_extraction_methods(self):
        """Figure 2: Extraction Method Comparison"""
        file_path = self.results_dir / "experiment_2_extraction_methods.csv"
        if not file_path.exists():
            print(f"⊘ Skipping Experiment 2: {file_path} not found")
            return

        df = pd.read_csv(file_path)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Extraction Method Comparison", fontsize=16, fontweight="bold")

        # Aggregate by method
        method_stats = (
            df.groupby("extraction_method")
            .agg(
                {
                    "contextual_relevance": "mean",
                    "answer_completeness": "mean",
                    "num_facts_extracted": "mean",
                    "avg_inference_time": "mean",
                }
            )
            .reset_index()
        )

        methods = method_stats["extraction_method"].values
        x = np.arange(len(methods))
        width = 0.6

        # Plot 1: Quality Metrics
        ax = axes[0, 0]
        relevance = method_stats["contextual_relevance"].values
        completeness = method_stats["answer_completeness"].values

        x_pos = np.arange(len(methods))
        ax.bar(x_pos - width / 4, relevance, width / 2, label="Relevance", alpha=0.8)
        ax.bar(
            x_pos + width / 4, completeness, width / 2, label="Completeness", alpha=0.8
        )
        ax.set_xlabel("Extraction Method")
        ax.set_ylabel("Score")
        ax.set_title("Quality Metrics by Method")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Facts Extracted
        ax = axes[0, 1]
        facts = method_stats["num_facts_extracted"].values
        ax.bar(x, facts, width, color="coral", alpha=0.8)
        ax.set_xlabel("Extraction Method")
        ax.set_ylabel("Avg Facts Extracted")
        ax.set_title("Information Extraction Volume")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Inference Time
        ax = axes[1, 0]
        times = method_stats["avg_inference_time"].values
        ax.bar(x, times, width, color="green", alpha=0.7)
        ax.set_xlabel("Extraction Method")
        ax.set_ylabel("Avg Inference Time (s)")
        ax.set_title("Computational Efficiency")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Per-Dataset Performance
        ax = axes[1, 1]
        datasets = df["dataset"].unique()
        for i, dataset in enumerate(datasets):
            data = df[df["dataset"] == dataset]
            ax.plot(
                data["extraction_method"],
                data["contextual_relevance"],
                marker="o",
                label=dataset,
                linewidth=2,
            )
        ax.set_xlabel("Extraction Method")
        ax.set_ylabel("Contextual Relevance")
        ax.set_title("Performance by Dataset")
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "fig2_extraction_methods.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_experiment_3_bandwidth_savings(self):
        """Figure 3: Bandwidth Savings Analysis"""
        file_path = self.results_dir / "experiment_3_bandwidth_savings.csv"
        if not file_path.exists():
            print(f"⊘ Skipping Experiment 3: {file_path} not found")
            return

        df = pd.read_csv(file_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Bandwidth Savings Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Compression vs Layer Selection
        ax = axes[0]
        for dataset in df["dataset"].unique():
            data = df[df["dataset"] == dataset]
            ax.plot(
                data["layer_selection_ratio"] * 100,
                data["compression_ratio"],
                marker="o",
                label=dataset,
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("Layer Selection Ratio (%)")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Compression vs Layer Selection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Bandwidth Saved
        ax = axes[1]
        ratio_groups = df.groupby("layer_selection_ratio")[
            "total_bandwidth_saved_mb"
        ].sum()
        ax.plot(
            ratio_groups.index * 100,
            ratio_groups.values,
            marker="s",
            linewidth=3,
            markersize=10,
            color="darkgreen",
        )
        ax.fill_between(ratio_groups.index * 100, 0, ratio_groups.values, alpha=0.3)
        ax.set_xlabel("Layer Selection Ratio (%)")
        ax.set_ylabel("Total Bandwidth Saved (MB)")
        ax.set_title("Cumulative Bandwidth Savings")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "fig3_bandwidth_savings.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_experiment_4_scalability(self):
        """Figure 4: Scalability Study"""
        file_path = self.results_dir / "experiment_4_scalability.csv"
        if not file_path.exists():
            print(f"⊘ Skipping Experiment 4: {file_path} not found")
            return

        df = pd.read_csv(file_path)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Scalability Across Model Sizes", fontsize=16, fontweight="bold")

        # Aggregate by model size
        model_stats = (
            df.groupby("model_size")
            .agg(
                {
                    "contextual_relevance": "mean",
                    "compression_ratio": "mean",
                    "avg_inference_time": "mean",
                }
            )
            .reset_index()
        )

        model_sizes = model_stats["model_size"].values
        x = np.arange(len(model_sizes))

        # Plot 1: Quality
        ax = axes[0]
        relevance = model_stats["contextual_relevance"].values
        ax.bar(x, relevance, color="steelblue", alpha=0.8)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Contextual Relevance")
        ax.set_title("Quality Across Model Sizes")
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Compression
        ax = axes[1]
        compression = model_stats["compression_ratio"].values
        ax.bar(x, compression, color="coral", alpha=0.8)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Compression Efficiency")
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Inference Time
        ax = axes[2]
        times = model_stats["avg_inference_time"].values
        ax.bar(x, times, color="green", alpha=0.7)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Avg Inference Time (s)")
        ax.set_title("Computational Cost")
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = self.figures_dir / "fig4_scalability.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_experiment_5_realworld(self):
        """Figure 5: Real-world Scenarios"""
        file_path = self.results_dir / "experiment_5_realworld_scenarios.csv"
        if not file_path.exists():
            print(f"⊘ Skipping Experiment 5: {file_path} not found")
            return

        df = pd.read_csv(file_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Real-world Scenario Performance", fontsize=16, fontweight="bold")

        scenarios = df["scenario"].unique()

        # Plot 1: Quality Comparison
        ax = axes[0]
        x = np.arange(len(scenarios))
        width = 0.25

        relevance = [
            df[df["scenario"] == s]["contextual_relevance"].mean() for s in scenarios
        ]
        completeness = [
            df[df["scenario"] == s]["answer_completeness"].mean() for s in scenarios
        ]

        ax.bar(x - width / 2, relevance, width, label="Relevance", alpha=0.8)
        ax.bar(x + width / 2, completeness, width, label="Completeness", alpha=0.8)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Score")
        ax.set_title("Quality Metrics by Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(["Conversational\nQA", "Multi-hop\nReasoning"], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Bandwidth Savings
        ax = axes[1]
        bandwidth = [
            df[df["scenario"] == s]["bandwidth_saved_mb"].sum() for s in scenarios
        ]
        compression = [
            df[df["scenario"] == s]["compression_ratio"].mean() for s in scenarios
        ]

        ax_twin = ax.twinx()

        bars = ax.bar(
            x, bandwidth, color="steelblue", alpha=0.7, label="Bandwidth Saved"
        )
        line = ax_twin.plot(
            x, compression, "ro-", linewidth=2, markersize=10, label="Compression Ratio"
        )

        ax.set_xlabel("Scenario")
        ax.set_ylabel("Bandwidth Saved (MB)", color="steelblue")
        ax_twin.set_ylabel("Compression Ratio", color="red")
        ax.set_title("Efficiency Metrics by Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(["Conversational\nQA", "Multi-hop\nReasoning"], rotation=0)
        ax.grid(True, alpha=0.3, axis="y")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        output_file = self.figures_dir / "fig5_realworld_scenarios.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def create_summary_table(self):
        """Create summary table for paper"""
        print("\n📊 Creating summary table...")

        # Try to load all experiment results
        summary_data = []

        for i in range(1, 6):
            pattern = f"experiment_{i}_*.csv"
            files = list(self.results_dir.glob(pattern))

            for file in files:
                df = pd.read_csv(file)
                if not df.empty:
                    summary_data.append(
                        {
                            "Experiment": f"Exp {i}",
                            "File": file.name,
                            "Samples": len(df),
                            "Avg Relevance": df.get(
                                "contextual_relevance", pd.Series([0])
                            ).mean(),
                            "Avg Compression": df.get(
                                "compression_ratio", pd.Series([0])
                            ).mean(),
                        }
                    )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            output_file = self.results_dir / "summary_table.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"✓ Saved summary table: {output_file}")
            print("\n" + summary_df.to_string(index=False))
        else:
            print("⊘ No data to summarize")

    def generate_all_figures(self):
        """Generate all publication figures"""
        print("🎨 Generating all publication figures...\n")

        self.plot_experiment_1_compression_quality()
        self.plot_experiment_2_extraction_methods()
        self.plot_experiment_3_bandwidth_savings()
        self.plot_experiment_4_scalability()
        self.plot_experiment_5_realworld()
        self.create_summary_table()

        print("\n✅ All figures generated!")
        print(f"📁 Saved to: {self.figures_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiment_results",
        help="Directory containing experiment results",
    )

    args = parser.parse_args()

    visualizer = ExperimentVisualizer(results_dir=args.results_dir)
    visualizer.generate_all_figures()


if __name__ == "__main__":
    main()
