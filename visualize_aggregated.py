"""
visualize_aggregated.py
-----------------------
Load `aggregated_metrics.json` and plot mean ± std percent changes for key metrics.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = 'analysis_plots'
os.makedirs(OUT_DIR, exist_ok=True)

with open('aggregated_metrics.json', 'r') as f:
    agg = json.load(f)

keys = list(agg['comparison'].keys())
means = [agg['comparison'][k]['mean_pct_change'] or 0.0 for k in keys]
stds = [agg['comparison'][k]['std_pct_change'] or 0.0 for k in keys]

x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(max(8, len(keys)), 5))
ax.bar(x, means, yerr=stds, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels([k.replace('_',' ') for k in keys], rotation=45, ha='right')
ax.set_ylabel('Percent change (NL -> CVL)')
ax.set_title('Aggregated percent change (mean ± std) across runs')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'aggregated_percent_change.png'), dpi=150)
print('Saved aggregated plot to', os.path.join(OUT_DIR, 'aggregated_percent_change.png'))
