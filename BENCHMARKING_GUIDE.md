# CVL Benchmarking Guide

## Overview

This comprehensive benchmarking suite tests two major features of the CVL (Compressed Vector Language) system:

1. **Compressed Vector Language (CVL)** - Efficient vector-based agent communication
2. **Truth Tokens / Honesty Tokens** - Verifiable honesty scoring for agent messages

## Quick Start

### Run Full Benchmarks (Recommended)
```bash
python run_benchmarks.py
```
**Time:** ~3-5 minutes  
**Sample Size:** 1000 messages  
**Output:** `cvl_benchmark_results.json`

### Custom Configuration
```bash
# Use specific number of samples
python run_benchmarks.py --samples 500

# Save to custom file
python run_benchmarks.py --output my_results.json

# Show help
python run_benchmarks.py --help
```

## What Gets Benchmarked

### 1. Compression Efficiency (`benchmark_compression_ratio`)
- **Compression ratio** (e.g., 20x, 30x)
- **Space savings** percentage
- **Compression speed** (milliseconds per message)
- **Decompression speed** (milliseconds per message)
- **Throughput** (messages per second)
- **Success rate** (percentage of successful compressions)

**Target Metrics:**
- Compression ratio: ≥ 20x
- Compression time: < 5ms
- Success rate: > 95%

### 2. Semantic Preservation (`benchmark_semantic_preservation`)
- **Cosine similarity** between original and decompressed embeddings
- **Message type preservation** accuracy
- **Priority preservation** accuracy
- **Overall semantic score** (composite metric)
- **Embedding distance** (Euclidean)

**Target Metrics:**
- Cosine similarity: ≥ 0.85
- Type preservation: > 90%
- Overall semantic score: ≥ 0.80

### 3. Task-Specific Performance (10 Tasks)

Tests CVL compression on 10 different task types:

1. **Arithmetic** - Math word problems (addition, subtraction, multiplication, division)
2. **Summarization** - Condensing passages into summaries
3. **Paraphrasing** - Rewriting sentences in different words
4. **Sentence Completion** - Cloze tests / fill-in-the-blank
5. **Classification** - Sentiment analysis (positive/negative/neutral)
6. **Translation** - Multi-language translation tasks
7. **QA Factual** - Factual question answering
8. **Commonsense** - Common sense reasoning questions
9. **Analogies** - Word relationship analogies (A:B::C:?)
10. **Entity Extraction** - Named entity recognition (NER)

**Metrics per Task:**
- Compression ratio
- Accuracy rate (preserved message type)
- Average compression time

**Target Metrics:**
- Task accuracy: > 85%
- Consistent compression across tasks

### 4. Truth Token System (`benchmark_truth_tokens`)

Tests the honesty/truth token implementation:

- **Honesty score calculation** (0.0 to 1.0)
  - Based on uncertainty markers
  - Confidence calibration
  - Message clarity
  - Specificity indicators

- **Verification system**
  - Hash verification (cryptographic commitment)
  - Confidence calibration checking
  - Outcome prediction accuracy

- **Challenge mechanism**
  - Agent-to-agent challenges
  - Challenge validity rate
  - False challenge detection

- **Reputation tracking**
  - Per-agent reputation scores
  - System-wide honesty metrics
  - Challenge history

**Key Metrics:**
- Average honesty score
- Honesty rate (% of honest messages)
- Challenge success rate
- Verification accuracy

**Target Metrics:**
- Avg honesty score: ≥ 0.70
- Honesty rate: > 75%
- Challenge accuracy: > 60%

### 5. Truth Token + CVL Integration

Tests integration of truth tokens with compressed messages:

- Total message size (compressed + truth token)
- Truth scalar preservation
- Integration success rate
- End-to-end workflow

**Target Metrics:**
- Integration success: > 95%
- Total size: < 15 bytes (8 bytes CVL + 4 bytes truth token)

### 6. Overall System Evaluation

Calculates composite score (0-100) with letter grade:

**Score Composition:**
- Compression performance: 30%
- Semantic preservation: 25%
- Task performance: 25%
- Truth token system: 20%

**Grading Scale:**
- A (93-100): Excellent, production-ready
- B (80-92): Good, minor improvements needed
- C (70-79): Satisfactory, optimization recommended
- D (60-69): Fair, significant improvements needed
- F (<60): Needs major work

## File Structure

```
compressed-vector-language-al/
├── truth_token_system.py      # Truth token implementation (300+ lines)
├── task_datasets.py            # 10 task dataset generators (500+ lines)
├── cvl_benchmark_suite.py      # Main benchmark framework (600+ lines)
├── run_benchmarks.py           # Easy execution script (200+ lines)
├── unsupervised_cvl.py         # Core CVL system (existing)
├── real_data_generator.py      # Agent data generator (existing)
└── BENCHMARKING_GUIDE.md       # This file
```

## Understanding the Output

### Console Output

The benchmark suite provides real-time progress updates:

```
[BENCHMARK 1/6] COMPRESSION RATIO & PERFORMANCE
  Progress: 20/100...
  Progress: 40/100...
  ...
  ✓ Compression Ratio: 24.5x
  ✓ Space Savings: 95.9%
  ✓ Avg Compression Time: 3.45ms

[BENCHMARK 2/6] SEMANTIC PRESERVATION
  ...
```

### JSON Output File

Results are saved to `cvl_benchmark_results.json`:

```json
{
  "compression_benchmarks": {
    "compression_ratio": 24.5,
    "space_savings_percent": 95.9,
    ...
  },
  "task_benchmarks": {
    "arithmetic": {...},
    "summarization": {...},
    ...
  },
  "truth_token_benchmarks": {
    "avg_honesty_score": 0.756,
    "honesty_rate": 0.82,
    ...
  },
  "overall_metrics": {
    "overall_cvl_score": 87.3,
    "letter_grade": "B+",
    ...
  }
}
```

## Truth Token Details

### How Truth Tokens Work

1. **Creation**: Agent creates message with self-assessed confidence
2. **Honesty Calculation**: System analyzes message for honesty indicators
3. **Encoding**: Truth token encoded as scalar (0.0-1.0) appended to vector
4. **Transmission**: Compressed message + truth token sent (total ~12 bytes)
5. **Verification**: Receiving agent can verify honesty and challenge if needed
6. **Reputation**: System tracks agent reputation over time

### Honesty Score Factors

**Increases Honesty Score:**
- Uncertainty markers + low confidence (honest about uncertainty)
- No uncertainty + high confidence (clear and confident)
- Specific details (numbers, names, dates)
- Appropriate message length (5-50 words)

**Decreases Honesty Score:**
- Uncertainty markers + high confidence (contradiction)
- Very short + very confident (suspicious)
- Excessive hedging words
- Overly verbose messages

### Challenge Mechanism

Agents can challenge each other's messages:

```python
# Agent B challenges Agent A's message
challenge_result = truth_system.challenge_agent(
    challenger_id="agent_B",
    challenged_token=agent_a_token,
    evidence={"contradiction_score": 0.85}
)

if challenge_result['challenge_valid']:
    # Challenge successful - Agent A loses reputation
else:
    # Challenge rejected - Agent B might be penalized
```

## Task Dataset Details

### Dataset Statistics

Each task type generates samples with:
- Question/input text
- Expected answer/output
- Metadata (difficulty, category, etc.)
- Task-specific fields

**Sample Distribution (30 samples per task):**
- Arithmetic: 30 problems
- Summarization: 30 passages
- Paraphrasing: 30 sentence pairs
- Sentence Completion: 30 cloze tests
- Classification: 30 sentiment examples
- Translation: 30 translation pairs
- QA Factual: 30 factual questions
- Commonsense: 30 reasoning questions
- Analogies: 30 word relationships
- Entity Extraction: 30 NER examples

**Total: 300 samples across 10 task types**

## Interpreting Results

### Excellent Performance (Score ≥ 85)
✅ System is production-ready  
✅ All components working well  
✅ No immediate action needed  

### Good Performance (Score 75-84)
✓ System working well overall  
→ Minor optimizations recommended  
→ Check specific component scores for improvement areas  

### Satisfactory Performance (Score 65-74)
⚠️ System functional but needs improvement  
→ Focus on lowest-scoring components  
→ Consider parameter tuning  

### Needs Improvement (Score < 65)
❌ Significant issues detected  
→ Review compression algorithms  
→ Check truth token thresholds  
→ Validate semantic preservation  

## Common Issues & Solutions

### Low Compression Ratio
**Problem:** Compression ratio < 15x  
**Solutions:**
- Increase PCA components
- Optimize vector quantization
- Check codebook size

### Poor Semantic Preservation
**Problem:** Cosine similarity < 0.75  
**Solutions:**
- Increase compressed dimensions
- Adjust PCA explained variance threshold
- Use better sentence encoder

### Low Truth Token Honesty Rate
**Problem:** Honesty rate < 60%  
**Solutions:**
- Adjust verification threshold (default: 0.68)
- Retune honesty calculation weights
- Check confidence calibration logic

### Slow Compression
**Problem:** Compression time > 10ms  
**Solutions:**
- Optimize vector quantization lookup
- Cache sentence embeddings
- Use batch processing

## Advanced Usage

### Running Individual Benchmarks

```python
from cvl_benchmark_suite import CVLBenchmarkSuite
from unsupervised_cvl import UnsupervisedCVL

# Initialize
cvl = UnsupervisedCVL()
suite = CVLBenchmarkSuite(cvl)

# Run individual benchmarks
compression_results = suite.benchmark_compression_ratio(messages)
semantic_results = suite.benchmark_semantic_preservation(messages)
truth_results = suite.benchmark_truth_tokens(messages)
```

### Custom Task Datasets

```python
from task_datasets import TaskDatasetGenerator

# Generate specific task
generator = TaskDatasetGenerator()
arithmetic_data = generator.generate_arithmetic(num_samples=100)
summarization_data = generator.generate_summarization(num_samples=50)

# Save custom dataset
generator.save_datasets("my_custom_dataset.json")
```

### Truth Token Demonstration

```python
from truth_token_system import demonstrate_truth_tokens

# Run interactive demonstration
demonstrate_truth_tokens()
```


## Troubleshooting

### Import Errors
```bash
# Install required packages
pip install -r requirements.txt
```

### Memory Issues
```bash
# Use quick mode for less memory usage
python run_benchmarks.py --quick --samples 200
```

### Slow Execution
```bash
# Reduce sample size
python run_benchmarks.py --samples 300
```

