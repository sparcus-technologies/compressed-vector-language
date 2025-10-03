"""
run_experiments.py
------------------
Run multiple trials of the NL vs CVL comparison to compute mean ± std for key metrics.
Saves `all_runs.json` and `aggregated_metrics.json`.
"""

import json
import os
import random
from statistics import mean, stdev

from run_metrics_comparison import run_metrics_comparison

OUT_ALL = 'all_runs.json'
OUT_AGG = 'aggregated_metrics.json'

def set_seed(s):
    random.seed(s)

def aggregate(runs):
    # runs: list of dicts {'nl_metrics','cvl_metrics','comparison'}
    keys = list(runs[0]['comparison'].keys())
    agg = {'comparison': {}}
    for k in keys:
        vals = [r['comparison'][k]['Pct_Change'] for r in runs if isinstance(r['comparison'][k].get('Pct_Change'), (int, float))]
        if vals:
            agg['comparison'][k] = {'mean_pct_change': mean(vals), 'std_pct_change': stdev(vals) if len(vals) > 1 else 0.0}
        else:
            agg['comparison'][k] = {'mean_pct_change': None, 'std_pct_change': None}
    return agg

if __name__ == '__main__':
    NUM_RUNS = 10
    results = []
    for i in range(NUM_RUNS):
        seed = 1000 + i
        set_seed(seed)
        print(f"Running trial {i+1}/{NUM_RUNS} (seed={seed})")
        metrics = run_metrics_comparison(num_messages=50, desired_agents=10, time_spacing=0.05, return_metrics=True)
        metrics['seed'] = seed
        results.append(metrics)
        # save incremental
        with open(OUT_ALL, 'w') as f:
            json.dump(results, f, indent=2)

    agg = aggregate(results)
    with open(OUT_AGG, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"Saved {len(results)} runs to {OUT_ALL} and aggregated metrics to {OUT_AGG}")
