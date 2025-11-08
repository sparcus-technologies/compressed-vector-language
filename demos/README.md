# Q-KVComm Demos

Quick-start demonstrations of Q-KVComm's core capabilities.

## Available Demos

### 1. **basic_demo.py** - Getting Started

The simplest introduction to Q-KVComm. Shows:

- Loading models
- Setting up Q-KVComm configuration
- Calibration process
- Basic context-query communication
- Compression metrics

**Run it:**

```bash
python demos/basic_demo.py
```

**Best for:** First-time users wanting to understand the basics.

---

### 2. **cross_model_demo.py** - Heterogeneous Models

Demonstrates KV cache transfer between DIFFERENT model architectures. Shows:

- Using different sender and receiver models (Qwen2.5-1.5B → TinyLlama-1.1B)
- Scalar vs per-dimension calibration
- Cross-architecture compatibility
- Multiple test cases

**Run it:**

```bash
python demos/cross_model_demo.py
```

**Best for:** Understanding how Q-KVComm handles different model architectures.

---

### 3. **quantization_demo.py** - Compression Levels

Shows how different quantization levels affect compression and quality. Demonstrates:

- Testing multiple bit-widths (4-bit, 6-bit, 8-bit)
- Compression vs quality trade-offs
- Adaptive quantization
- Visual comparison of results

**Run it:**

```bash
python demos/quantization_demo.py
```

**Best for:** Understanding compression trade-offs and choosing optimal settings.

---

### 4. **extraction_demo.py** - Adaptive Information Extraction

Shows Q-KVComm's intelligent information extraction capabilities. Demonstrates:

- Extracting key facts from long contexts
- Different extraction methods (simple, YAKE, SpaCy NER, hybrid)
- Selective KV cache transmission based on relevance
- Comparison: no extraction vs compression only vs full Q-KVComm

**Run it:**

```bash
python demos/extraction_demo.py
```

**Best for:** Understanding how extraction helps with long documents and targeted queries.

---

### 5. **multi_agent_demo.py** - Agent Communication

Real-world multi-agent scenario where agents share information efficiently. Shows:

- Specialist agent (processes data) → Responder agent (answers queries)
- Knowledge transfer without re-processing
- Memory savings in multi-agent systems
- Practical use case

**Run it:**

```bash
python demos/multi_agent_demo.py
```

**Best for:** Seeing Q-KVComm in a realistic multi-agent application.

---

## Quick Start

1. **Start simple:** Run `basic_demo.py` first
2. **Explore compression:** Try `quantization_demo.py` to understand settings
3. **Try extraction:** Run `extraction_demo.py` to see adaptive information extraction
4. **Advanced features:** Check `cross_model_demo.py` and `multi_agent_demo.py`

## Feature Coverage

The demos cover Q-KVComm's core capabilities:

- **KV Cache Compression** (`basic_demo.py`, `quantization_demo.py`) - Quantized KV cache transfer
- **Information Extraction** (`extraction_demo.py`) - Adaptive fact extraction from long contexts
- **Cross-Architecture** (`cross_model_demo.py`) - Different sender/receiver models
- **Multi-Agent Systems** (`multi_agent_demo.py`) - Practical agent communication

## Requirements

All demos require:

- PyTorch
- Transformers
- Q-KVComm package

Some demos may download models automatically (~3-6 GB).

## Comparison with Traditional Approaches

For a detailed comparison of Q-KVComm vs traditional natural language communication, see the experiments folder:

```bash
python experiments_new/exp1_compression_quality.py
```
