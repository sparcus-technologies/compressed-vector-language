# compressed-vector-language
Compressed Vector Language (CVL) for Agent Talk


Branch summary:
# compressed-vector-language
Compressed Vector Language (CVL) for Agent Talk

This repository implements an unsupervised compression pipeline (CVL) for agent-to-agent communication, compares compressed communication to natural language (NL) exchanges, and provides metrics + visualizations to evaluate trade-offs (bandwidth, memory, coordination, precision).

The version in this workspace includes small research enhancements (task-aware objective, compositional/codebook quantizers) and an experiments harness to run multiple trials and produce aggregated statistics.

What you'll find here
- End-to-end demo and experiment scripts that generate synthetic agent messages and compare NL vs CVL.
- A small unsupervised CVL implementation: sentence embeddings → PCA → vector quantization → compact encodings.
- Metrics to evaluate communication (token usage, coordination, latency, precision) and visualization scripts that save PNG plots under `analysis_plots/`.

Files & roles
- `real_data_generator.py` — synthesize multi-agent message streams. Supports `agent_count` and `time_spacing` so messages have agent IDs and timestamps (used by coordination metrics).
- `unsupervised_cvl.py` — core CVL implementation. Embeddings (SentenceTransformer) → adaptive PCA → simple vector quantizer (compositional codebooks) → compress/decompress routines.
- `metrics.py` — computes NL and CVL metrics (token usage, communication overhead, coordination scores, soft evaluation, partial success rate, etc.). Includes helpers to make cosine/embedding-based comparisons robust to small datasets.
- `run_metrics_comparison.py` — single-run pipeline: generate data, fit CVL, compress/decompress, compute metrics, and write outputs (`nl_metrics.json`, `cvl_metrics.json`, `comparison.json`, and `analysis.md`). Can return metrics programmatically for programmatic experiments.
- `visualize_metrics.py` — produce per-run comparison plots (saved to `analysis_plots/`): memory/speed, loss/precision, coordination plots.
- `run_experiments.py` — multi-run harness that calls the comparison pipeline across several random seeds, saves `all_runs.json` (per-run metrics) and `aggregated_metrics.json` (mean ± std across runs).
- `visualize_aggregated.py` — plots aggregated percent-change (mean ± std) across runs and saves `analysis_plots/aggregated_percent_change.png`.
- `demo_unsupervised.py` — small demo to exercise the unsupervised CVL pipeline end-to-end interactively.

How the pipeline works (high level)
1. Data generation: `real_data_generator.py` creates synthetic agent messages. Each message contains agent id, content, timestamp, and a content length/token estimate used for token-usage metrics.
2. Embeddings: `unsupervised_cvl.py` creates semantic embeddings for each message using a SentenceTransformer model (`all-MiniLM-L6-v2` by default).
3. Dimensionality reduction: PCA is applied; the implementation chooses a safe number of components automatically (min of requested dim, sample count, and feature dims) to avoid errors on small datasets.
4. Quantization / Codebooks: a simple compositional quantizer learns multiple small codebooks and represents each compressed vector as a compact index tuple (the code is intentionally small — research-enhanced settings use tiny encodings).
5. Encode / Decode: messages are compressed into compact CVL representations and can be decompressed back into approximate vectors for evaluation.
6. Metrics: `metrics.py` computes token usage (NL tokens vs CVL bytes), coordination/latency metrics (that rely on agent ids and timestamps), soft evaluation scores (cosine similarity between NL embeddings and decompressed embeddings), and other task-level signals.
7. Visualization and analysis: per-run visualizers create PNG plots, the multi-run harness aggregates results across seeds, and `visualize_aggregated.py` plots mean ± std percent-changes.

Experiments we ran
- Single-run comparison: `python run_metrics_comparison.py` — generates one dataset run, fits CVL, computes metrics, and writes JSON outputs + `analysis.md` with a local summary.
- Multi-run aggregation: `python run_experiments.py` — runs the single-run pipeline across multiple seeds (default: 10), saving per-run results as `all_runs.json` and aggregated statistics to `aggregated_metrics.json`. We then plotted aggregated percent-change via `python visualize_aggregated.py`.

From two agents to multi-agent
- Original demos assumed a minimal agent setup. To make coordination metrics meaningful we updated the data generator and pipeline:
	- `real_data_generator.generate_dataset(...)` now accepts `agent_count` and `time_spacing` and rotates token authorship across multiple agent IDs and assigns spaced timestamps.
	- `run_metrics_comparison.py` and the metric functions were updated to expect and use `agent_id` and `timestamp` for each message when computing coordination and real-time metrics.
	- This lets us measure multi-agent coordination effectiveness, observation sharing, real-time coordination, and message round counts in a realistic way rather than relying on a fixed two-agent toy.

Key findings (aggregated)
- Token usage / memory: CVL reduced token-equivalent usage by approximately 74.87% (mean percent change = 74.8737%, std = 0.6719%). In practice this means CVL achieves a ~3.9× reduction in communicated token volume for the synthetic tasks we ran.
- Computational memory (proxy): shows the same ~74.87% mean reduction (same values as token usage in the aggregated outputs), indicating CVL's compact representations are much smaller to store/transmit.
- Soft evaluation score: reported a large mean percent change (mean = 4837.69%) with very high variance (std ≈ 27590.42%). This indicates the embedding-space comparison between NL and decompressed vectors is noisy and sensitive in our current setup (small sample size, embedding dimension adjustments, and padding/truncation heuristics can amplify percent-change). We recommend treating this particular value cautiously and running more trials or refining the soft-evaluation metric before drawing firm conclusions.
- Coordination & other task metrics: most coordination-level metrics (coordination effectiveness, partial success rate, realtime coordination score, etc.) showed near-zero mean percent change in the current aggregated run. That suggests compressed communication preserved the task-level coordination signals in these synthetic tasks, but it should be validated on more complex tasks and larger datasets.

Notes and caveats
- The dataset used for these runs was synthetic (50 messages per run by default). Results will change with larger datasets, different message types, or different task objectives.
- The large variance in the soft-evaluation metric needs investigation — possible causes include embedding dimension mismatch after PCA / quantization, numerical scaling, or unstable cosine estimates for very short vectors.
- The CVL implementation here is intentionally simple and designed for experiments rather than production. It uses a SentenceTransformer for semantic embeddings and a small PCA + compositional quantizer.

How to reproduce
1. Create a Python venv and install dependencies:
```powershell
python -m venv venv; .\\venv\\Scripts\\Activate.ps1; pip install -r requirements.txt
```
2. Single-run comparison (writes JSON + `analysis.md` and plots):
```powershell
python run_metrics_comparison.py
python visualize_metrics.py
```
3. Multi-run aggregation (default 10 trials) and aggregated plot:
```powershell
python run_experiments.py
python visualize_aggregated.py
```

Where outputs land
- JSON metrics: `nl_metrics.json`, `cvl_metrics.json`, `comparison.json` (per-run)
- Multi-run outputs: `all_runs.json`, `aggregated_metrics.json`
- Plots: `analysis_plots/` contains `memory_speed_comparison.png`, `loss_precision_comparison.png`, `coordination_comparison.png`, and `aggregated_percent_change.png`.

Next steps
- Increase dataset size and agent complexity to stress-test coordination metrics.
- Stabilize and unit-test the soft-evaluation metric to reduce sensitivity.
- Add CI tests for metrics and small regression tests for the compressor/decompressor.

If you'd like, I can commit and push these experiment outputs and this README to your `fardinCode` branch, or add a short section to `analysis.md` that embeds the generated images automatically.

---
