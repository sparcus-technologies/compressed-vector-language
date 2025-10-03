"""
visualize_metrics.py
--------------------
Load the JSON metric outputs produced by `run_metrics_comparison.py` and
create simple comparison plots (NL vs CVL) to highlight differences.

Saves PNG files under `analysis_plots/`.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = 'analysis_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_bar_comparison(metric_keys, nl_metrics, cvl_metrics, title, filename):
    nl_vals = [nl_metrics.get(k, 0) for k in metric_keys]
    cvl_vals = [cvl_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(metric_keys)), 4))
    ax.bar(x - width/2, nl_vals, width, label='NL')
    ax.bar(x + width/2, cvl_vals, width, label='CVL')

    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == '__main__':
    nl_metrics = load_json('nl_metrics.json')
    cvl_metrics = load_json('cvl_metrics.json')

    # Select some numeric metrics to compare
    keys = [
        'token_usage', 'communication_overhead', 'computational_efficiency_memory',
        'acl', 'number_of_message_passing_rounds', 'replanning_speed', 'task_duration'
    ]

    # Some metrics may be nested or not numeric; coerce to float when possible
    def to_number(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    nl_metrics_num = {k: to_number(v) for k, v in nl_metrics.items()}
    cvl_metrics_num = {k: to_number(v) for k, v in cvl_metrics.items()}

    plot_bar_comparison(keys, nl_metrics_num, cvl_metrics_num, 'Memory & Speed Metrics (NL vs CVL)', 'memory_speed_comparison.png')

    # Precision / loss metrics
    keys2 = ['partial_success_rate', 'soft_evaluation_scores', 'damage_minimization', 'average_mistakes']
    plot_bar_comparison(keys2, nl_metrics_num, cvl_metrics_num, 'Loss & Precision Metrics (NL vs CVL)', 'loss_precision_comparison.png')

    # Coordination metrics
    keys3 = ['coordination_effectiveness', 'observation_sharing_score', 'realtime_coordination_score']
    plot_bar_comparison(keys3, nl_metrics_num, cvl_metrics_num, 'Coordination Metrics (NL vs CVL)', 'coordination_comparison.png')
