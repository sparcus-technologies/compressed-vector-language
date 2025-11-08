# Q-KVComm Research Experiments

Publication-quality experiments for evaluating Q-KVComm performance.

## Overview

This folder contains individual experiments designed for research evaluation and publication. Each experiment can be run independently and generates CSV results with detailed metrics.

## Experiment Files

### **exp1_compression_quality.py**

Tests different compression levels and measures quality preservation.

**What it does:**

- Tests 4-bit, 6-bit, 8-bit, and 12-bit quantization
- Measures contextual relevance, answer completeness, semantic fidelity
- Calculates compression ratios and bandwidth savings
- Runs on SQuAD, HotpotQA, and NarrativeQA datasets

**Usage:**

```bash
python experiments_new/exp1_compression_quality.py --max-samples 100 --device cuda
```

**Output:** `experiment_results/experiment_1_compression_quality.csv`

---

### **exp2_extraction_methods.py**

Ablation study comparing different extraction methods.

**What it does:**

- Compares: simple, YAKE, SpaCy NER, and hybrid extraction
- Measures quality, efficiency, and facts extracted
- Tests on SQuAD, HotpotQA, CoQA, NarrativeQA

**Usage:**

```bash
python experiments_new/exp2_extraction_methods.py --max-samples 100 --device cuda
```

**Output:** `experiment_results/experiment_2_extraction_methods.csv`

---

### **exp3_bandwidth_savings.py**

Quantifies bandwidth reduction with different layer selection ratios.

**What it does:**

- Tests 50%, 70%, 90%, and 100% layer selection
- Measures bandwidth saved vs quality preservation
- Shows compression-quality trade-offs

**Usage:**

```bash
python experiments_new/exp3_bandwidth_savings.py --max-samples 100 --device cuda
```

**Output:** `experiment_results/experiment_3_bandwidth_savings.csv`

---

### **exp4_scalability.py**

Tests Q-KVComm across different model sizes.

**What it does:**

- Evaluates with TinyLlama-1.1B and Qwen2.5-1.5B
- Measures scalability with increasing model parameters
- Compares performance and efficiency

**Usage:**

```bash
python experiments_new/exp4_scalability.py --max-samples 100 --device cuda
```

**Output:** `experiment_results/experiment_4_scalability.csv`

---

### **exp5_realworld_scenarios.py**

Tests Q-KVComm in practical multi-agent scenarios.

**What it does:**

- Scenario 1: Conversational QA with caching
- Scenario 2: Multi-hop reasoning
- Real-world use case evaluation

**Usage:**

```bash
python experiments_new/exp5_realworld_scenarios.py --max-samples 100 --device cuda
```

**Output:** `experiment_results/experiment_5_realworld_scenarios.csv`

---

## Supporting Files

### **benchmark_suite.py**

Core benchmarking infrastructure used by all experiments. Provides:

- Dataset loaders (SQuAD, HotpotQA, CoQA, NarrativeQA, Natural Questions)
- Agentic communication metrics evaluation
- Semantic similarity analysis
- Result aggregation and reporting

**Note:** This is automatically imported by experiment scripts - you don't run it directly.

### **experiment_base.py**

Base class providing common functionality for all experiments:

- Device management (auto-detection of CUDA)
- Random seed setting for reproducibility
- Output directory creation
- Consistent formatting utilities

### **visualization.py**

Post-processing tool for creating publication-quality plots from experiment CSV files.

---

## Visualization

After running experiments, generate publication-quality figures:

```bash
python experiments_new/visualization.py --results-dir experiment_results
```

This creates plots in `experiment_results/figures/`:

- Compression vs quality trade-offs
- Extraction method comparisons
- Bandwidth savings analysis
- Scalability curves
- Real-world scenario results

## Common Arguments

All experiments support:

- `--output-dir PATH` - Where to save results (default: `experiment_results`)
- `--device {auto,cuda,cpu}` - Device to use (default: `auto`)
- `--max-samples N` - Max samples per dataset (default: 100)
- `--model NAME` - Model to use (default: TinyLlama-1.1B-Chat-v1.0)
- `--seed N` - Random seed for reproducibility (default: 42)

## Running All Experiments

To run a complete evaluation suite:

```bash
# Run each experiment
python experiments_new/exp1_compression_quality.py --max-samples 50
python experiments_new/exp2_extraction_methods.py --max-samples 50
python experiments_new/exp3_bandwidth_savings.py --max-samples 50
python experiments_new/exp4_scalability.py --max-samples 50
python experiments_new/exp5_realworld_scenarios.py --max-samples 50

# Generate visualizations
python experiments_new/visualization.py
```

## Results Structure

```
experiment_results/
├── experiment_1_compression_quality.csv
├── experiment_2_extraction_methods.csv
├── experiment_3_bandwidth_savings.csv
├── experiment_4_scalability.csv
├── experiment_5_realworld_scenarios.csv
├── figures/
│   ├── fig1_compression_quality.png
│   ├── fig2_extraction_methods.png
│   ├── fig3_bandwidth_savings.png
│   ├── fig4_scalability.png
│   └── fig5_realworld.png
└── exp1_bits_*/  # Detailed per-run results
```

## Key Metrics

All experiments measure:

- **Contextual Relevance** - How well context is preserved
- **Answer Completeness** - Quality of generated answers
- **Compression Ratio** - How much data is saved
- **Bandwidth Saved (MB)** - Actual bytes saved
- **Inference Time** - Processing speed

## Note on visualization.py

The `visualization.py` script is specifically for creating publication-ready plots from experiment CSV files. It's not a demo - it's a post-processing tool for researchers preparing papers or presentations.

**When to use it:**

- After running experiments
- When preparing research papers
- For creating presentation figures
- To analyze trends across experiments
