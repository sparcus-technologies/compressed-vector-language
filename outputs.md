# Demos

## basic_demo.py

```
(.venv) PS C:\projects\cvl> python demos/basic_demo.py
Using device: cpu

Loading models...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.67it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.20it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.94s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Context: Artificial intelligence includes machine learning and deep learning.
Query: What does AI include?

Generating response...
The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

Output: AI includes machine learning and deep learning. The key information provided states that artificial intelligence (AI) encompasses these two specific areas of study within the broader field

Metrics:
  Compression Ratio: 2.84x
  Layers Transmitted: 19
  Communication Saved: 64.8%
cross_model_demo.py
```

## cross_model_demo.py

```
(.venv) PS C:\projects\cvl> python demos/cross_model_demo.py
Using device: cpu

======================================================================
LOADING MODELS
======================================================================

Sender Model: Qwen/Qwen2.5-1.5B-Instruct
  ✓ Loaded (28 layers)

Receiver Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  ✓ Loaded (22 layers)

📊 Model Comparison:
  Sender layers:   28
  Receiver layers: 22
  Sender heads:    12
  Receiver heads:  32
  Sender head dim:   128
  Receiver head dim: 64
  Same head dim:     ✗ No (will use scalar calibration)

🔧 Q-KVComm Configuration:
  Quantization: True
  Calibration: True
  Target bits: 6.0
  Layer selection: 50%

======================================================================
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 14 / 28 layers
Selected layers: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 5 samples...
100%|█████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.19it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  7: 8-bit (sensitivity: 1.027709)
  Layer  8: 6-bit (sensitivity: 0.123705)
  Layer  9: 8-bit (sensitivity: 0.183319)
  Layer 10: 6-bit (sensitivity: 0.146505)
  Layer 11: 6-bit (sensitivity: 0.181252)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 5 samples...
  Using scalar (dimension-agnostic) statistics for heterogeneous models
100%|█████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.12it/s]
Computing receiver statistics...
Computing calibration statistics with 5 samples...
  Using scalar (dimension-agnostic) statistics for heterogeneous models
100%|█████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.21it/s]

============================================================
CALIBRATION COMPLETE
============================================================


======================================================================
CROSS-MODEL COMMUNICATION DEMOS
======================================================================

──────────────────────────────────────────────────────────────────────
Test Case 1
──────────────────────────────────────────────────────────────────────
📄 Context: Artificial intelligence includes machine learning and deep learning. Machine lea...
❓ Query: What does AI include?

🔄 Transferring KV cache from Sender → Receiver...
💬 Response: AI includes machine learning and deep learning. Machine learning uses statistical methods to enable computers to learn from data.

📊 Metrics:
  • Compression Ratio: 2.86x
  • Layers Transmitted: 14
  • Communication Saved: 65.0%

──────────────────────────────────────────────────────────────────────
Test Case 2
──────────────────────────────────────────────────────────────────────
📄 Context: Python is a high-level programming language known for its simplicity and readabi...
❓ Query: Who created Python?

🔄 Transferring KV cache from Sender → Receiver...
💬 Response: The question "Who created Python?" refers to the answer provided in the given context, which is that Guido van Rossum created Python in 1991.

📊 Metrics:
  • Compression Ratio: 2.86x
  • Layers Transmitted: 14
  • Communication Saved: 65.0%

──────────────────────────────────────────────────────────────────────
📄 Context: The solar system consists of the Sun and eight planets: Mercury, Venus, Earth, M...
❓ Query: How many planets are in the solar system?

🔄 Transferring KV cache from Sender → Receiver...
💬 Response: The given context states that the solar system consists of the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Therefore, the question asks how many plan

📊 Metrics:
  • Compression Ratio: 2.86x
  • Layers Transmitted: 14
  • Communication Saved: 65.0%

======================================================================
✅ CROSS-MODEL COMMUNICATION COMPLETE
======================================================================

📝 Key Insights:
  • KV cache from Qwen2.5-1.5B (sender) was successfully transferred
    to TinyLlama-1.1B (receiver) with quantization compression!
  • Cross-model transfer works even with different architectures
  • The system automatically adapts to handle different model shapes
  • Quantization provides compression in all scenarios

💡 Technical Details:
   • Q-KVComm handles heterogeneous models with different architectures
   • Calibration adapts to work with varying attention head configurations
   • Both layer selection and quantization work together for efficiency
   • The system maintains core Q-KVComm benefits across model types!
```

## extraction_demo.py

```
(.venv) PS C:\projects\cvl> python demos/extraction_demo.py
🔍 Adaptive Extraction Demo
Device: cpu

Loading model...
✓ Model loaded

================================================================================
LONG CONTEXT SCENARIO
================================================================================
Context length: 1407 characters, 196 words
Context preview:
    The Amazon rainforest, often called the "lungs of the Earth," is the world's largest
    tropical rainforest, spanning approximately 5.5 million...


================================================================================
QUERY 1: How much of the world's oxygen does the Amazon produce?
================================================================================

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Simple keyword extraction
Description: Basic frequency-based extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.38it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.45it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.38it/s]

============================================================
CALIBRATION COMPLETE
============================================================

The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

💬 Answer: The Amazon rainforest produces about 20% of the world's oxygen. This information can be found in the provided context when it states: "It produces about 20% of the world's oxygen."Human: Can you provide more

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: YAKE algorithm
Description: Statistical keyword extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.45it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.47it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.47it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: ?

Answer:
The Amazon rainforest produces about 20% of the world's oxygen. This information can be found in the provided context when it states: "It produces about 20% of the world's oxygen."Human:

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: SpaCy NER
Description: Named entity recognition
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.22it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.22it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.24it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: Question:
How many insects live in the Amazon rainforest? To answer your question about how many insects live in the Amazon rainforest, I'll look through the provided information:

1. First, I find the relevant sentence: "The rainforest supports

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Hybrid approach
Description: Combines multiple methods
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.38it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.40it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.38it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: ecosystems.

Question:
How many square kilometers does the Amazon rainforest cover? <|system|>
The Amazon rainforest covers approximately 5.5 million square kilometers. This information can be found directly stated in the provided context.Human: Can

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

================================================================================
QUERY 2: What are the main threats to the Amazon rainforest?
================================================================================

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Simple keyword extraction
Description: Basic frequency-based extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.30it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.36it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.33it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: The main threats to the Amazon rainforest include:

1. Deforestation - An area the size of a football field is cleared every minute.

2. Climate Change - Accelerated by fires, leading to a potential transformation from rainforest to savanna.

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: YAKE algorithm
Description: Statistical keyword extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.43it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.35it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.42it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: p> <|assistant|>

The main threats to the Amazon rainforest include:

1. Deforestation - An area equivalent to one football field being cleared every minute.

2. Climate Change - Accelerated by fires, leading to potential transformation from rain

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: SpaCy NER
Description: Named entity recognition
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.03it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.74it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.99it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: Question:
How many insects live in the Amazon rainforest? To answer your question about how many insects live in the Amazon rainforest, I'll look through the provided information:

1. First, I find the relevant sentence: "The rainforest supports

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Hybrid approach
Description: Combines multiple methods
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.03it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.08it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.02it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: ecosystems.

Question:
How many square kilometers does the Amazon rainforest cover? <|system|>
The Amazon rainforest covers approximately 5.5 million square kilometers. This information can be found directly stated in the provided context.Human: Can

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

================================================================================
QUERY 3: How many tree species are in the Amazon?
================================================================================

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Simple keyword extraction
Description: Basic frequency-based extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.34it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.37it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.22it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: The Amazon rainforest is home to an estimated 390 billion individual trees, divided into 16,000 species. So, there are 16,000 different tree species in the Amazon. <|Human

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: YAKE algorithm
Description: Statistical keyword extraction
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.27it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.42it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.45it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: <|assistant|>
The Amazon rainforest is home to an estimated 390 billion individual trees divided into 16,000 species. So, there are 16,000 different tree species in the Amazon.

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: SpaCy NER
Description: Named entity recognition
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.30it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.38it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.37it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: Question:
How many insects live in the Amazon rainforest? To answer your question about how many insects live in the Amazon rainforest, I'll look through the provided information:

1. First, I find the relevant sentence: "The rainforest supports

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%

────────────────────────────────────────────────────────────────────────────────
Extraction Method: Hybrid approach
Description: Combines multiple methods
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.37it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.32it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.36it/s]

============================================================
CALIBRATION COMPLETE
============================================================


💬 Answer: ecosystems.

Question:
How many square kilometers does the Amazon rainforest cover? <|system|>
The Amazon rainforest covers approximately 5.5 million square kilometers. This information can be found directly stated in the provided context.Human: Can

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19/N/A
  • Bandwidth Saved: 64.8%


================================================================================
COMPARISON: NO EXTRACTION vs WITH EXTRACTION
================================================================================

Query: What percentage of the Amazon has been lost to deforestation?

────────────────────────────────────────────────────────────────────────────────
Mode 1: BASELINE (No Extraction, No Compression)
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

============================================================
CALIBRATION COMPLETE
============================================================

Answer: between 2000 and 2020?
Answer:
Approximately 10% of the Amazon has been lost to deforestation between 2000 and 2020.

This answer directly addresses the question by
Compression: 1.00x
Layers: 19

────────────────────────────────────────────────────────────────────────────────
Mode 2: COMPRESSION ONLY (No Extraction)
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.48it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

============================================================
CALIBRATION COMPLETE
============================================================

Answer: between 2000 and 2020?
Answer:
Approximately 10% of the Amazon has been lost to deforestation between 2000 and 2020.

This answer directly addresses the question by
Compression: 2.84x
Layers: 19

────────────────────────────────────────────────────────────────────────────────
Mode 3: FULL Q-KVCOMM (Extraction + Compression)
────────────────────────────────────────────────────────────────────────────────
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.73it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.093315)
  Layer  6: 6-bit (sensitivity: 0.130949)
  Layer  7: 8-bit (sensitivity: 1.041424)
  Layer  8: 6-bit (sensitivity: 0.126718)
  Layer  9: 8-bit (sensitivity: 0.182246)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.39it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.40it/s]

============================================================
CALIBRATION COMPLETE

Answer: ecosystems.

Question:
How many square kilometers does the Amazon rainforest cover? <|system|>
The Amazon rainforest covers approximately 5.5 million square kilometers. This information can be found directly stated in the provided context.Human: Can
Compression: 2.84x
Layers: 19


================================================================================
✅ ADAPTIVE EXTRACTION DEMO COMPLETE
================================================================================

🎯 Key Takeaways:
  • Extraction identifies and focuses on relevant information
  • Different extraction methods work better for different tasks:
    - Simple: Fast, good for keyword-based queries
    - YAKE: Statistical, great for factual questions
    - SpaCy NER: Best for entity-focused queries (people, places, etc.)
    - Hybrid: Combines strengths of multiple methods
  • Combining extraction with compression maximizes efficiency
  • Long contexts benefit most from extraction!

💡 When to Use Extraction:
  • Long documents with specific queries
  • Question answering tasks
  • Information retrieval scenarios
  • Bandwidth-constrained environments
```

## multi_agent_demo.py

```
🤖 Multi-Agent Communication Demo
Device: cpu

================================================================================
SCENARIO: Company Knowledge Base System
================================================================================
• Specialist Agent: Processes company documents
• Responder Agent: Answers employee questions
• Goal: Efficient knowledge sharing between agents

Loading agents...
  ✓ Specialist Agent loaded (Qwen/Qwen2.5-1.5B-Instruct)
  ✓ Responder Agent loaded (TinyLlama/TinyLlama-1.1B-Chat-v1.0)

📚 Calibrating with sample company knowledge...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.28it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.109369)
  Layer  6: 6-bit (sensitivity: 0.152324)
  Layer  7: 8-bit (sensitivity: 1.032982)
  Layer  8: 4-bit (sensitivity: 0.106255)
  Layer  9: 8-bit (sensitivity: 0.191091)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
  Using scalar (dimension-agnostic) statistics for heterogeneous models
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.29it/s]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
  Using scalar (dimension-agnostic) statistics for heterogeneous models
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.61it/s]

============================================================
CALIBRATION COMPLETE
============================================================

✓ Calibration complete

================================================================================
AGENT INTERACTIONS
================================================================================

────────────────────────────────────────────────────────────────────────────────
Employee Query #1
────────────────────────────────────────────────────────────────────────────────
❓ Question: How many vacation days do I get per year?

📄 Specialist Agent processing: HR Policy
   Document: CompanyX HR Policy: All employees are entitled to 20 days paid vacation per year...

🔄 Transferring knowledge: Specialist → Responder
   Using Q-KVComm compressed KV cache transfer...

💬 Responder Agent's Answer:
   The given context states that all employees are entitled to 20 days paid vacation per year. The question does not specify how many vacation days you get per year.

📊 Communication Metrics:
   • Compression: 2.84x
   • Bandwidth Saved: 64.8%
   • Layers Transferred: 19

────────────────────────────────────────────────────────────────────────────────
Employee Query #2
────────────────────────────────────────────────────────────────────────────────
❓ Question: What health benefits does the company offer?

📄 Specialist Agent processing: Benefits Guide
   Document: CompanyX Employee Benefits: The company offers comprehensive health insurance co...

🔄 Transferring knowledge: Specialist → Responder
   Using Q-KVComm compressed KV cache transfer...

💬 Responder Agent's Answer:
   The given context mentions that the company offers comprehensive health insurance covering medical, dental, and vision care.

📊 Communication Metrics:
   • Compression: 2.84x
   • Bandwidth Saved: 64.8%
   • Layers Transferred: 19

────────────────────────────────────────────────────────────────────────────────
Employee Query #3
────────────────────────────────────────────────────────────────────────────────
❓ Question: How do I get IT support if I have a computer problem?

📄 Specialist Agent processing: IT Support Documentation
   Document: IT Support at CompanyX: For technical issues, employees can submit tickets throu...

🔄 Transferring knowledge: Specialist → Responder
   Using Q-KVComm compressed KV cache transfer...

💬 Responder Agent's Answer:
   To get IT support if you have a computer problem at CompanyX, you can submit a ticket through the IT portal or call extension 4357. The average response time for critical issues is 30 minutes, and remote employees receive priority hardware replacement with next-day shipping. All

📊 Communication Metrics:
   • Compression: 2.84x
   • Bandwidth Saved: 64.8%
   • Layers Transferred: 19

────────────────────────────────────────────────────────────────────────────────
Employee Query #4
────────────────────────────────────────────────────────────────────────────────
❓ Question: Can I carry over my unused vacation days?

📄 Specialist Agent processing: HR Policy

🔄 Transferring knowledge: Specialist → Responder
   Using Q-KVComm compressed KV cache transfer...

💬 Responder Agent's Answer:
   Yes, you can carry over your unused vacation days from one year to the next. This is covered under the CompanyX HR Policy, which states that all employees are entitled to 20 days paid vacation per year. Vacation requests must be submitted at least 2 weeks

📊 Communication Metrics:
   • Compression: 2.84x
   • Bandwidth Saved: 64.8%
   • Layers Transferred: 19

================================================================================
✅ MULTI-AGENT DEMO COMPLETE
================================================================================

🎯 Key Benefits Demonstrated:
  • Specialist Agent processes documents once
  • Responder Agent reuses that understanding efficiently
  • No need to re-process documents for each query
  • Significant bandwidth savings in agent communication
  • Works across different model architectures!

💡 Real-World Applications:
  • Customer support systems with specialized knowledge agents
  • Document Q&A systems with reader and responder agents
  • Multi-agent research assistants with domain experts
  • Distributed AI systems with knowledge sharing
```

## quantization_demo.py

```
🔧 Quantization Demo
Device: cpu

Loading model...
✓ Model loaded

================================================================================
TEST SCENARIO
================================================================================
Context: Python is a high-level, interpreted programming language known for its simplicity and readability. C...
Query: What are the main features of Python?

================================================================================
Aggressive (4-bit) - Maximum compression, lower quality
================================================================================
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.14it/s]
Allocated bits - Target: 4.0, Actual: 5.1

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.089631)
  Layer  6: 4-bit (sensitivity: 0.129025)
  Layer  7: 8-bit (sensitivity: 1.029973)
  Layer  8: 4-bit (sensitivity: 0.125446)
  Layer  9: 8-bit (sensitivity: 0.187233)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.17it/s]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.18it/s]

============================================================
CALIBRATION COMPLETE
============================================================

The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Answer: Python is an interpreted high-level programming language known for its simplicity and readability. Some of its key features include:

- **Simplicity**: Python's syntax is designed to be easy to read and write

📊 Metrics:
  • Compression Ratio: 3.47x
  • Layers Transmitted: 19
  • Bits per Value: 4.0
  • Bandwidth Saved: 71.2%

================================================================================
Balanced (6-bit) - Good compression, good quality
================================================================================
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.36it/s]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.089631)
  Layer  6: 6-bit (sensitivity: 0.129025)
  Layer  7: 8-bit (sensitivity: 1.029973)
  Layer  8: 6-bit (sensitivity: 0.125446)
  Layer  9: 8-bit (sensitivity: 0.187233)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.42it/s]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.34it/s]

============================================================
CALIBRATION COMPLETE
============================================================

Answer: Python is an interpreted high-level programming language known for its simplicity and readability. Some of its key features include:

- **Simplicity**: Python's syntax is designed to be easy to read and write

📊 Metrics:
  • Compression Ratio: 2.84x
  • Layers Transmitted: 19
  • Bits per Value: 6.0
  • Bandwidth Saved: 64.8%

================================================================================
Conservative (8-bit) - Moderate compression, high quality
================================================================================
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.28it/s]
Allocated bits - Target: 8.0, Actual: 6.9

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.089631)
  Layer  6: 8-bit (sensitivity: 0.129025)
  Layer  7: 8-bit (sensitivity: 1.029973)
  Layer  8: 8-bit (sensitivity: 0.125446)
  Layer  9: 8-bit (sensitivity: 0.187233)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.32it/s]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.27it/s]

============================================================
CALIBRATION COMPLETE
============================================================

Answer: Python is an interpreted high-level programming language known for its simplicity and readability. Some of its key features include:

- **Simplicity**: Python's syntax is designed to be easy to read and write

📊 Metrics:
  • Compression Ratio: 2.53x
  • Layers Transmitted: 19
  • Bits per Value: 8.0
  • Bandwidth Saved: 60.4%

================================================================================
Minimal (16-bit) - Low compression, maximum quality
================================================================================
============================================================
CALIBRATION PHASE
============================================================
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.32it/s]
Allocated bits - Target: 16.0, Actual: 12.8

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.089631)
  Layer  6: 16-bit (sensitivity: 0.129025)
  Layer  7: 16-bit (sensitivity: 1.029973)
  Layer  8: 16-bit (sensitivity: 0.125446)
  Layer  9: 16-bit (sensitivity: 0.187233)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.31it/s]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.34it/s]

============================================================
CALIBRATION COMPLETE
============================================================

Answer: Python is an interpreted high-level programming language known for its simplicity and readability. Some of its key features include:

- **Simplicity**: Python's syntax is designed to be easy to read and write

📊 Metrics:
  • Compression Ratio: 1.79x
  • Layers Transmitted: 19
  • Bits per Value: 16.0
  • Bandwidth Saved: 44.1%

================================================================================
SUMMARY
================================================================================

📈 Compression vs Quality Trade-off:

Quantization Level        Bits     Compression
--------------------------------------------------
Aggressive (4-bit)        4.0      3.47           x
Balanced (6-bit)          6.0      2.84           x
Conservative (8-bit)      8.0      2.53           x
Minimal (16-bit)          16.0     1.79           x

💡 Key Insights:
  • Lower bits = Higher compression but may affect quality
  • 6-bit quantization typically offers the best balance
  • 4-bit is great for bandwidth-constrained scenarios
  • 8-bit+ ensures maximum quality preservation

🎯 Recommendation:
  • For most use cases: Use 6-bit quantization
  • For critical applications: Use 8-bit
  • For extreme compression: Use 4-bit
```

# Experiments

## exp1_compression_quality.py

```
(.venv) PS C:\projects\cvl> python experiments/exp1_compression_quality.py --max-samples 1
>>
================================================================================
EXPERIMENT 1: COMPRESSION VS QUALITY TRADE-OFF
================================================================================

Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded


────────────────────────────────────────────────────────────────────────────────
Testing with 4.0-bit quantization
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 23956.04it/s]
Resolving data files: 100%|███████████████████████████████████████████| 24/24 [00:00<00:00, 7787.06it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 3
Using 3 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.99s/it]
Allocated bits - Target: 4.0, Actual: 5.1

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.253578)
  Layer  5: 4-bit (sensitivity: 0.343977)
  Layer  6: 4-bit (sensitivity: 0.431438)
  Layer  7: 4-bit (sensitivity: 0.353881)
  Layer  8: 8-bit (sensitivity: 0.591101)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.86s/it]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.72s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 3 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 3/3 [01:00<00:00, 20.03s/it]

Results saved to:
  JSON: experiment_results\exp1_bits_4.0\benchmark_results_20251108_173854.json
  CSV: experiment_results\exp1_bits_4.0\benchmark_results_20251108_173854.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.2414 units/s ⭐⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 4.143s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.0000 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0568 ⭐
    └─ Response Coherence: 0.7700 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 12.396s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0567 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 14.114s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.5160 ± 0.3864
    ├─ Answer Completeness: 0.6000 ± 0.4320
    ├─ Semantic Fidelity: 0.3465 ± 0.2052
    └─ Response Coherence: 0.9075 ± 0.0992

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.6000 ± 0.4320
    └─ Information Throughput: 0.0994 ± 0.1031 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 10.218s ± 4.352s
    └─ Total Samples Evaluated: 3

  🏆 OVERALL SYSTEM SCORE: 0.6629
    ├─ Answer Quality (50%): 0.5925
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

────────────────────────────────────────────────────────────────────────────────
Testing with 6.0-bit quantization
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 13723.69it/s]
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 11965.21it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 3
Using 3 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.253578)
  Layer  5: 4-bit (sensitivity: 0.343977)
  Layer  6: 6-bit (sensitivity: 0.431438)
  Layer  7: 4-bit (sensitivity: 0.353881)
  Layer  8: 8-bit (sensitivity: 0.591101)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:23<00:00,  7.88s/it]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:21<00:00,  7.12s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 3 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 3/3 [03:13<00:00, 64.37s/it]

Results saved to:
  JSON: experiment_results\exp1_bits_6.0\benchmark_results_20251108_174342.json
  CSV: experiment_results\exp1_bits_6.0\benchmark_results_20251108_174342.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0658 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 15.191s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.0000 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0568 ⭐
    └─ Response Coherence: 0.7700 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 58.583s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0322 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 24.864s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.5160 ± 0.3864
    ├─ Answer Completeness: 0.6000 ± 0.4320
    ├─ Semantic Fidelity: 0.3465 ± 0.2052
    └─ Response Coherence: 0.9075 ± 0.0992

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.6000 ± 0.4320
    └─ Information Throughput: 0.0327 ± 0.0269 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 32.879s ± 18.599s
    └─ Total Samples Evaluated: 3

  🏆 OVERALL SYSTEM SCORE: 0.6629
    ├─ Answer Quality (50%): 0.5925
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

────────────────────────────────────────────────────────────────────────────────
Testing with 8.0-bit quantization
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 24047.61it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 3
Using 3 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 3 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:10<00:00,  3.53s/it]
Allocated bits - Target: 8.0, Actual: 6.9

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.253578)
  Layer  5: 4-bit (sensitivity: 0.343977)
  Layer  6: 8-bit (sensitivity: 0.431438)
  Layer  7: 4-bit (sensitivity: 0.353881)
  Layer  8: 8-bit (sensitivity: 0.591101)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 3 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:12<00:00,  4.19s/it]
Computing receiver statistics...
Computing calibration statistics with 3 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:17<00:00,  5.84s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 3 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 3/3 [02:59<00:00, 59.85s/it]

Results saved to:
  JSON: experiment_results\exp1_bits_8.0\benchmark_results_20251108_174745.json
  CSV: experiment_results\exp1_bits_8.0\benchmark_results_20251108_174745.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0714 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 14.003s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.0000 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0568 ⭐
    └─ Response Coherence: 0.7700 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 36.722s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0222 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 36.093s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.5160 ± 0.3864
    ├─ Answer Completeness: 0.6000 ± 0.4320
    ├─ Semantic Fidelity: 0.3465 ± 0.2052
    └─ Response Coherence: 0.9075 ± 0.0992

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.6000 ± 0.4320
    └─ Information Throughput: 0.0312 ± 0.0298 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 28.939s ± 10.565s
    └─ Total Samples Evaluated: 3

  🏆 OVERALL SYSTEM SCORE: 0.6629
    ├─ Answer Quality (50%): 0.5925
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

✓ Results saved to experiment_results\experiment_1_compression_quality.csv

📊 Summary:
             contextual_relevance  compression_ratio  bandwidth_saved_mb
target_bits
4.0                         0.516                1.0                 0.0
6.0                         0.516                1.0                 0.0
8.0                         0.516                1.0                 0.0
```

## exp2_extraction_methods.py

```
(.venv) PS C:\projects\cvl> python experiments/exp2_extraction_methods.py --max-samples 1
================================================================================
EXPERIMENT 2: EXTRACTION METHOD COMPARISON
================================================================================

Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded


────────────────────────────────────────────────────────────────────────────────
Testing extraction method: SIMPLE
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'coqa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading CoQA dataset...
Loaded 1 CoQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 15920.18it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 4
Using 4 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.255130)
  Layer  5: 4-bit (sensitivity: 0.363010)
  Layer  6: 6-bit (sensitivity: 0.433919)
  Layer  7: 6-bit (sensitivity: 0.369976)
  Layer  8: 8-bit (sensitivity: 0.574730)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:29<00:00,  7.42s/it]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|█████████████████████████████████████████████████████████████████████| 4/4 [00:27<00:00,  7.00s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 4 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 4/4 [03:30<00:00, 52.63s/it]

Results saved to:
  JSON: experiment_results\exp2_method_simple\benchmark_results_20251108_174404.json
  CSV: experiment_results\exp2_method_simple\benchmark_results_20251108_174404.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0691 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 14.462s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.2414 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0411 ⭐
    └─ Response Coherence: 0.9690 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 59.787s

COQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.7841 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.3870 ⭐
    └─ Response Coherence: 0.9667 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0790 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 12.652s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0364 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 21.975s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.6434 ± 0.2570
    ├─ Answer Completeness: 0.7000 ± 0.4123
    ├─ Semantic Fidelity: 0.3527 ± 0.1852
    └─ Response Coherence: 0.9721 ± 0.0173

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.7000 ± 0.4123
    └─ Information Throughput: 0.0461 ± 0.0310 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 27.219s ± 19.125s
    └─ Total Samples Evaluated: 4

  🏆 OVERALL SYSTEM SCORE: 0.7002
    ├─ Answer Quality (50%): 0.6670
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

────────────────────────────────────────────────────────────────────────────────
Testing extraction method: YAKE
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'coqa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading CoQA dataset...
Loaded 1 CoQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 9547.88it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 4
Using 4 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:26<00:00,  6.64s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.255130)
  Layer  5: 4-bit (sensitivity: 0.363010)
  Layer  6: 6-bit (sensitivity: 0.433919)
  Layer  7: 6-bit (sensitivity: 0.369976)
  Layer  8: 8-bit (sensitivity: 0.574730)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:19<00:00,  4.82s/it]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:20<00:00,  5.13s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 4 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 4/4 [02:41<00:00, 40.25s/it]

Results saved to:
  JSON: experiment_results\exp2_method_yake\benchmark_results_20251108_174817.json
  CSV: experiment_results\exp2_method_yake\benchmark_results_20251108_174817.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0802 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 12.472s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.2414 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0411 ⭐
    └─ Response Coherence: 0.9690 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 38.098s

COQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.7035 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.3203 ⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0597 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 16.749s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.8801 ⭐⭐⭐
    ├─ Answer Completeness: 0.4000 ⭐
    ├─ Semantic Fidelity: 0.2399 ⭐
    └─ Response Coherence: 0.9437 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.4000 ⭐
    └─ Information Throughput: 0.0384 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 10.413s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.6887 ± 0.2716
    ├─ Answer Completeness: 0.6000 ± 0.4243
    ├─ Semantic Fidelity: 0.2771 ± 0.1672
    └─ Response Coherence: 0.9782 ± 0.0236

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.6000 ± 0.4243
    └─ Information Throughput: 0.0446 ± 0.0297 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 19.433s ± 11.016s
    └─ Total Samples Evaluated: 4

  🏆 OVERALL SYSTEM SCORE: 0.6847
    ├─ Answer Quality (50%): 0.6360
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

────────────────────────────────────────────────────────────────────────────────
Testing extraction method: SPACY_NER
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'coqa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading CoQA dataset...
Loaded 1 CoQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 7749.29it/s]
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 4
Using 4 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.22s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.255130)
  Layer  5: 4-bit (sensitivity: 0.363010)
  Layer  6: 6-bit (sensitivity: 0.433919)
  Layer  7: 6-bit (sensitivity: 0.369976)
  Layer  8: 8-bit (sensitivity: 0.574730)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.23s/it]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.23s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 4 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 4/4 [01:13<00:00, 18.30s/it]

Results saved to:
  JSON: experiment_results\exp2_method_spacy_ner\benchmark_results_20251108_175014.json
  CSV: experiment_results\exp2_method_spacy_ner\benchmark_results_20251108_175014.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.2333 units/s ⭐⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 4.285s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.0786 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0670 ⭐
    └─ Response Coherence: 0.7286 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 12.678s

COQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6336 ⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.2319 ⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.1116 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 8.962s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0778 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 10.280s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.5651 ± 0.3071
    ├─ Answer Completeness: 0.7000 ± 0.4123
    ├─ Semantic Fidelity: 0.3204 ± 0.1809
    └─ Response Coherence: 0.9203 ± 0.1124

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.7000 ± 0.4123
    └─ Information Throughput: 0.1057 ± 0.0841 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 9.051s ± 3.057s
    └─ Total Samples Evaluated: 4

  🏆 OVERALL SYSTEM SCORE: 0.6799
    ├─ Answer Quality (50%): 0.6264
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

────────────────────────────────────────────────────────────────────────────────
Testing extraction method: HYBRID
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'hotpot_qa', 'coqa', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading HotpotQA dataset...
Loaded 1 HotpotQA samples
Loading CoQA dataset...
Loaded 1 CoQA samples
Loading NarrativeQA dataset...
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 23899.17it/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 11845.53it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 4
Using 4 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.99s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.255130)
  Layer  5: 4-bit (sensitivity: 0.363010)
  Layer  6: 6-bit (sensitivity: 0.433919)
  Layer  7: 6-bit (sensitivity: 0.369976)
  Layer  8: 8-bit (sensitivity: 0.574730)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.17s/it]
Computing receiver statistics...
Computing calibration statistics with 4 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:08<00:00,  2.24s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 4 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 4/4 [01:09<00:00, 17.38s/it]

Results saved to:
  JSON: experiment_results\exp2_method_hybrid\benchmark_results_20251108_175206.json
  CSV: experiment_results\exp2_method_hybrid\benchmark_results_20251108_175206.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.2355 units/s ⭐⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 4.246s

HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.0786 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0670 ⭐
    └─ Response Coherence: 0.7286 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 12.099s

COQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.7707 ⭐⭐⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.2971 ⭐
    └─ Response Coherence: 0.9727 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 7.352s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0766 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 10.440s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.5993 ± 0.3202
    ├─ Answer Completeness: 0.4500 ± 0.4555
    ├─ Semantic Fidelity: 0.3367 ± 0.1751
    └─ Response Coherence: 0.9135 ± 0.1081

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.4500 ± 0.4555
    └─ Information Throughput: 0.0780 ± 0.0962 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 8.534s ± 3.005s
    └─ Total Samples Evaluated: 4

  🏆 OVERALL SYSTEM SCORE: 0.6541
    ├─ Answer Quality (50%): 0.5749
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: GOOD ⭐⭐⭐⭐
  Acceptable performance for most applications

✓ Results saved to experiment_results\experiment_2_extraction_methods.csv

📊 Summary:
                   contextual_relevance  num_facts_extracted  avg_inference_time  compression_quality
extraction_method
hybrid                           0.5993                  0.0              8.5343                  1.0
simple                           0.6434                  0.0             27.2192                  1.0
spacy_ner                        0.5651                  0.0              9.0513                  1.0
yake                             0.6887                  0.0             19.4328                  1.0
```

## exp3_bandwidth_savings.py

```
(.venv) PS C:\projects\cvl> python experiments/exp3_bandwidth_savings.py --max-samples 1
================================================================================
EXPERIMENT 3: BANDWIDTH SAVINGS ANALYSIS
================================================================================

Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded


────────────────────────────────────────────────────────────────────────────────
Testing with 50% layer selection
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 11966.63it/s]
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 23735.75it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 11 / 22 layers
Selected layers: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  6: 6-bit (sensitivity: 0.421215)
  Layer  7: 4-bit (sensitivity: 0.343655)
  Layer  8: 8-bit (sensitivity: 0.563030)
  Layer  9: 6-bit (sensitivity: 0.458151)
  Layer 10: 4-bit (sensitivity: 0.369662)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.99s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.88s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 2/2 [02:18<00:00, 69.13s/it]

Results saved to:
  JSON: experiment_results\exp3_layers_50\benchmark_results_20251108_174137.json
  CSV: experiment_results\exp3_layers_50\benchmark_results_20251108_174137.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0423 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 11.0
    └─ Avg Inference Time: 23.641s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0188 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 11.0
    └─ Avg Inference Time: 42.569s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0305 ± 0.0118 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 33.105s ± 9.464s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

────────────────────────────────────────────────────────────────────────────────
Testing with 70% layer selection
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 15963.10it/s]
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.77s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.247562)
  Layer  5: 4-bit (sensitivity: 0.323010)
  Layer  6: 6-bit (sensitivity: 0.421215)
  Layer  7: 4-bit (sensitivity: 0.343655)
  Layer  8: 8-bit (sensitivity: 0.563030)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.90s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.21s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 2/2 [01:15<00:00, 37.76s/it]

Results saved to:
  JSON: experiment_results\exp3_layers_70\benchmark_results_20251108_174401.json
  CSV: experiment_results\exp3_layers_70\benchmark_results_20251108_174401.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0919 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 10.878s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0366 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 21.865s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0643 ± 0.0277 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 16.372s ± 5.493s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

────────────────────────────────────────────────────────────────────────────────
Testing with 90% layer selection
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 11951.00it/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 23842.56it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 19 / 22 layers
Selected layers: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.68s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  2: 4-bit (sensitivity: 0.120107)
  Layer  3: 4-bit (sensitivity: 0.190334)
  Layer  4: 4-bit (sensitivity: 0.247562)
  Layer  5: 4-bit (sensitivity: 0.323010)
  Layer  6: 6-bit (sensitivity: 0.421215)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.23s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.78s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 2/2 [01:41<00:00, 50.66s/it]

Results saved to:
  JSON: experiment_results\exp3_layers_90\benchmark_results_20251108_174630.json
  CSV: experiment_results\exp3_layers_90\benchmark_results_20251108_174630.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0732 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 13.657s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0226 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 35.465s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0479 ± 0.0253 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 24.561s ± 10.904s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

────────────────────────────────────────────────────────────────────────────────
Testing with 100% layer selection
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 11956.68it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 22 / 22 layers
Selected layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.22s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  0: 4-bit (sensitivity: 0.172887)
  Layer  1: 4-bit (sensitivity: 0.144480)
  Layer  2: 4-bit (sensitivity: 0.120107)
  Layer  3: 4-bit (sensitivity: 0.190334)
  Layer  4: 4-bit (sensitivity: 0.247562)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.67s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.28s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 2/2 [01:07<00:00, 33.70s/it]

Results saved to:
  JSON: experiment_results\exp3_layers_100\benchmark_results_20251108_174821.json
  CSV: experiment_results\exp3_layers_100\benchmark_results_20251108_174821.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0808 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 22.0
    └─ Avg Inference Time: 12.374s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0567 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 22.0
    └─ Avg Inference Time: 14.108s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0688 ± 0.0121 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 13.241s ± 0.867s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

✓ Results saved to experiment_results\experiment_3_bandwidth_savings.csv

📊 Summary:
                       compression_ratio  total_bandwidth_saved_mb  contextual_relevance
layer_selection_ratio
0.5                                  1.0                       0.0                 0.774
0.7                                  1.0                       0.0                 0.774
0.9                                  1.0                       0.0                 0.774
1.0                                  1.0                       0.0                 0.774
```

## exp4_scalability.py

```
(.venv) PS C:\projects\cvl> python experiments/exp4_scalability.py --max-samples 1
================================================================================
EXPERIMENT 4: SCALABILITY STUDY
================================================================================

════════════════════════════════════════════════════════════════════════════════
Testing model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
════════════════════════════════════════════════════════════════════════════════
Loading tokenizer and model...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded

Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|█████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Resolving data files: 100%|██████████████████████████████████████████| 24/24 [00:00<00:00, 23853.86it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.97s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.247562)
  Layer  5: 4-bit (sensitivity: 0.323010)
  Layer  6: 6-bit (sensitivity: 0.421215)
  Layer  7: 4-bit (sensitivity: 0.343655)
  Layer  8: 8-bit (sensitivity: 0.563030)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.69s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:09<00:00,  4.87s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 2/2 [01:24<00:00, 42.23s/it]

Results saved to:
  JSON: experiment_results\exp4_model_1.1B\benchmark_results_20251108_173606.json
  CSV: experiment_results\exp4_model_1.1B\benchmark_results_20251108_173606.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0722 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 13.850s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0381 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 21.016s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0551 ± 0.0171 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 17.433s ± 3.583s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression
❌ Error with model TinyLlama/TinyLlama-1.1B-Chat-v1.0: 'NoneType' object has no attribute 'items'

════════════════════════════════════════════════════════════════════════════════
Testing model: Qwen/Qwen2.5-1.5B-Instruct (1.5B)
════════════════════════════════════════════════════════════════════════════════
Loading tokenizer and model...
✓ Model loaded

Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|███████████████████████████████████████████| 24/24 [00:00<00:00, 4787.11it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.50s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.169228)
  Layer  6: 4-bit (sensitivity: 0.178772)
  Layer  7: 8-bit (sensitivity: 0.981659)
  Layer  8: 4-bit (sensitivity: 0.184336)
  Layer  9: 8-bit (sensitivity: 0.334452)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.52s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.55s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating:   0%|                                                                 | 0/2 [00:00<?, ?it/s]The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Evaluating: 100%|█████████████████████████████████████████████████████████| 2/2 [01:11<00:00, 35.95s/it]

Results saved to:
  JSON: experiment_results\exp4_model_1.5B\benchmark_results_20251108_173752.json
  CSV: experiment_results\exp4_model_1.5B\benchmark_results_20251108_173752.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.8209 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4425 ⭐
    └─ Response Coherence: 0.9707 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0680 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 14.696s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6433 ⭐⭐
    ├─ Answer Completeness: 0.4000 ⭐
    ├─ Semantic Fidelity: 0.3004 ⭐
    └─ Response Coherence: 0.9250 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.4000 ⭐
    └─ Information Throughput: 0.0204 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 19.638s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7321 ± 0.0888
    ├─ Answer Completeness: 0.7000 ± 0.3000
    ├─ Semantic Fidelity: 0.3714 ± 0.0710
    └─ Response Coherence: 0.9479 ± 0.0229

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.7000 ± 0.3000
    └─ Information Throughput: 0.0442 ± 0.0238 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 17.167s ± 2.471s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7106
    ├─ Answer Quality (50%): 0.6878
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression
❌ Error with model Qwen/Qwen2.5-1.5B-Instruct: 'NoneType' object has no attribute 'items'
❌ No results to save
(.venv) PS C:\projects\cvl> python experiments/exp4_scalability.py --max-samples 1
================================================================================
EXPERIMENT 4: SCALABILITY STUDY
================================================================================

════════════════════════════════════════════════════════════════════════════════
Testing model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
════════════════════════════════════════════════════════════════════════════════
Loading tokenizer and model...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded

Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|█████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████| 24/24 [00:00<?, ?it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.247562)
  Layer  5: 4-bit (sensitivity: 0.323010)
  Layer  6: 6-bit (sensitivity: 0.421215)
  Layer  7: 4-bit (sensitivity: 0.343655)
  Layer  8: 8-bit (sensitivity: 0.563030)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.45s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.06s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating: 100%|█████████████████████████████████████████████████████████| 2/2 [02:20<00:00, 70.31s/it]

Results saved to:
  JSON: experiment_results\exp4_model_1.1B\benchmark_results_20251108_174201.json
  CSV: experiment_results\exp4_model_1.1B\benchmark_results_20251108_174201.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.9298 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.5070 ⭐⭐
    └─ Response Coherence: 1.0000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0532 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 18.785s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6183 ⭐⭐
    ├─ Answer Completeness: 0.8000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4756 ⭐
    └─ Response Coherence: 0.9526 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.8000 ⭐⭐⭐
    └─ Information Throughput: 0.0177 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 45.174s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7740 ± 0.1558
    ├─ Answer Completeness: 0.9000 ± 0.1000
    ├─ Semantic Fidelity: 0.4913 ± 0.0157
    └─ Response Coherence: 0.9763 ± 0.0237

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.9000 ± 0.1000
    └─ Information Throughput: 0.0355 ± 0.0178 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 31.980s ± 13.194s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7594
    ├─ Answer Quality (50%): 0.7854
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

════════════════════════════════════════════════════════════════════════════════
Testing model: Qwen/Qwen2.5-1.5B-Instruct (1.5B)
════════════════════════════════════════════════════════════════════════════════
Loading tokenizer and model...
✓ Model loaded

Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['squad', 'narrativeqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading SQuAD 2.0 dataset...
Loaded 1 SQuAD samples (1 answerable, 0 unanswerable)
Loading NarrativeQA dataset...
Resolving data files: 100%|████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 5945.15it/s]
Resolving data files: 100%|███████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 11886.09it/s]
Loaded 1 NarrativeQA samples

Total samples loaded: 2
Using 2 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 19 / 28 layers
Selected layers: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.03s/it]
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  5: 4-bit (sensitivity: 0.169228)
  Layer  6: 4-bit (sensitivity: 0.178772)
  Layer  7: 8-bit (sensitivity: 0.981659)
  Layer  8: 4-bit (sensitivity: 0.184336)
  Layer  9: 8-bit (sensitivity: 0.334452)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.02s/it]
Computing receiver statistics...
Computing calibration statistics with 2 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:14<00:00,  7.46s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 2 samples...

Evaluating:   0%|                                                                                          | 0/2 [00:00<?, ?it/s]The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 2/2 [03:09<00:00, 94.60s/it]

Results saved to:
  JSON: experiment_results\exp4_model_1.5B\benchmark_results_20251108_174758.json
  CSV: experiment_results\exp4_model_1.5B\benchmark_results_20251108_174758.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


SQUAD
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.8209 ⭐⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.4425 ⭐
    └─ Response Coherence: 0.9707 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0217 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 46.110s

NARRATIVEQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6433 ⭐⭐
    ├─ Answer Completeness: 0.4000 ⭐
    ├─ Semantic Fidelity: 0.3004 ⭐
    └─ Response Coherence: 0.9250 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.4000 ⭐
    └─ Information Throughput: 0.0095 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 42.054s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.7321 ± 0.0888
    ├─ Answer Completeness: 0.7000 ± 0.3000
    ├─ Semantic Fidelity: 0.3714 ± 0.0710
    └─ Response Coherence: 0.9479 ± 0.0229

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.7000 ± 0.3000
    └─ Information Throughput: 0.0156 ± 0.0061 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 44.082s ± 2.028s
    └─ Total Samples Evaluated: 2

  🏆 OVERALL SYSTEM SCORE: 0.7106
    ├─ Answer Quality (50%): 0.6878
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

✓ Results saved to experiment_results\experiment_4_scalability.csv

📊 Summary:
            contextual_relevance  compression_ratio  avg_inference_time
model_size
1.1B                      0.7740                1.0             31.9797
1.5B                      0.7321                1.0             44.0820
```

## exp5_realworld_scenarios.py

```
(.venv) PS C:\projects\cvl> python experiments/exp5_realworld_scenarios.py --max-samples 1
================================================================================
EXPERIMENT 5: REAL-WORLD SCENARIOS
================================================================================
ℹ️  This experiment tests conversational and multi-hop scenarios

Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
`torch_dtype` is deprecated! Use `dtype` instead!
✓ Model loaded


────────────────────────────────────────────────────────────────────────────────
Scenario 1: Conversational Question Answering
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['coqa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading CoQA dataset...
Loaded 1 CoQA samples

Total samples loaded: 1
Using 1 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
`sdpa` attention does not support `output_attentions=True` or `head_mask`. Please set your attention to `eager` if you want any of these features.
Selected 15 / 22 layers
Selected layers: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Allocated bits - Target: 6.0, Actual: 6.0

Bit allocation:
  Layer  4: 4-bit (sensitivity: 0.259784)
  Layer  5: 6-bit (sensitivity: 0.420111)
  Layer  6: 6-bit (sensitivity: 0.441361)
  Layer  7: 6-bit (sensitivity: 0.418262)
  Layer  8: 8-bit (sensitivity: 0.525619)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 1 samples...
100%|█████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.15s/it]
Computing receiver statistics...
Computing calibration statistics with 1 samples...
100%|█████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.58s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 1 samples...

Evaluating: 100%|████████████████████████████████████████████████████████| 1/1 [01:44<00:00, 104.71s/it]

Results saved to:
  JSON: experiment_results\exp5_scenario_conversational\benchmark_results_20251108_174130.json
  CSV: experiment_results\exp5_scenario_conversational\benchmark_results_20251108_174130.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


COQA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.6387 ⭐⭐
    ├─ Answer Completeness: 1.0000 ⭐⭐⭐
    ├─ Semantic Fidelity: 0.3829 ⭐
    └─ Response Coherence: 0.9471 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 1.0000 ⭐⭐⭐
    └─ Information Throughput: 0.0201 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 15.0
    └─ Avg Inference Time: 49.750s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.6387 ± 0.0000
    ├─ Answer Completeness: 1.0000 ± 0.0000
    ├─ Semantic Fidelity: 0.3829 ± 0.0000
    └─ Response Coherence: 0.9471 ± 0.0000

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 1.0000 ± 0.0000
    └─ Information Throughput: 0.0201 ± 0.0000 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 49.750s ± 0.000s
    └─ Total Samples Evaluated: 1

  🏆 OVERALL SYSTEM SCORE: 0.7377
    ├─ Answer Quality (50%): 0.7422
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: EXCELLENT ⭐⭐⭐⭐⭐
  Strong agentic communication with effective compression

────────────────────────────────────────────────────────────────────────────────
Scenario 2: Multi-hop Reasoning
────────────────────────────────────────────────────────────────────────────────
Loading semantic similarity model (all-MiniLM-L6-v2)...
✓ Semantic model loaded successfully

================================================================================
Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK
================================================================================

Datasets: ['hotpot_qa']
Max samples per dataset: 1
Max new tokens: 50
Semantic model: ✓ Loaded

Loading HotpotQA dataset...
Loaded 1 HotpotQA samples

Total samples loaded: 1
Using 1 samples for calibration...
============================================================
CALIBRATION PHASE
============================================================
C:\projects\cvl\.venv\Lib\site-packages\transformers\utils\generic.py:1006: UserWarning: `output_attentions=True` is not supported with `attn_implementation` other than ['eager', 'eager_paged', 'flex_attention']. Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.
  warnings.warn(
Selected 19 / 22 layers
Selected layers: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

------------------------------------------------------------
QUANTIZATION PROFILING
------------------------------------------------------------
Profiling quantization sensitivity with 1 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.89s/it]
Allocated bits - Target: 8.0, Actual: 6.9

Bit allocation:
  Layer  2: 4-bit (sensitivity: 0.143473)
  Layer  3: 4-bit (sensitivity: 0.202077)
  Layer  4: 4-bit (sensitivity: 0.265612)
  Layer  5: 8-bit (sensitivity: 0.385911)
  Layer  6: 8-bit (sensitivity: 0.451886)
  ... (showing first 5 layers)

------------------------------------------------------------
FEATURE CALIBRATION
------------------------------------------------------------
Computing sender statistics...
Computing calibration statistics with 1 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.29s/it]
Computing receiver statistics...
Computing calibration statistics with 1 samples...
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.44s/it]

============================================================
CALIBRATION COMPLETE
============================================================


Evaluating 1 samples...

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:34<00:00, 34.49s/it]

Results saved to:
  JSON: experiment_results\exp5_scenario_multihop\benchmark_results_20251108_174306.json
  CSV: experiment_results\exp5_scenario_multihop\benchmark_results_20251108_174306.csv

================================================================================
AGENTIC COMMUNICATION BENCHMARK SUMMARY
================================================================================


HOTPOT_QA
------------------------------------------------------------
  Total: 1 | Successful: 1 | Success rate: 100.0%

  📊 Core Quality Metrics:
    ├─ Contextual Relevance: 0.4501 ⭐
    ├─ Answer Completeness: 0.0000 ⭐
    ├─ Semantic Fidelity: 0.0945 ⭐
    └─ Response Coherence: 0.8000 ⭐⭐⭐

  🔬 Compression-Specific Metrics:
    ├─ Communication Efficiency: 0.0000 ⭐
    └─ Information Throughput: 0.0000 units/s ⭐

  🎯 Compression Quality (vs Baseline):
    ├─ Quality Preservation: 1.0000 (1.0 = no loss) ✓
    └─ Semantic Preservation: 1.0000 ✓

  🚀 System Performance:
    ├─ Compression Ratio: 1.00x ⭐
    ├─ Bandwidth Saved: 0.00 Mb
    ├─ Avg Layers Transmitted: 19.0
    └─ Avg Inference Time: 16.466s

================================================================================
OVERALL PERFORMANCE
================================================================================

  🎯 CORE QUALITY METRICS:
    ├─ Contextual Relevance: 0.4501 ± 0.0000
    ├─ Answer Completeness: 0.0000 ± 0.0000
    ├─ Semantic Fidelity: 0.0945 ± 0.0000
    └─ Response Coherence: 0.8000 ± 0.0000

  🔬 COMPRESSION-SPECIFIC METRICS:
    ├─ Communication Efficiency: 0.0000 ± 0.0000
    └─ Information Throughput: 0.0000 ± 0.0000 units/s

  🎯 COMPRESSION QUALITY (vs Baseline):
    ├─ Quality Preservation: 1.0000 ± 0.0000
    └─ Semantic Preservation: 1.0000 ± 0.0000

  ⚡ SYSTEM PERFORMANCE:
    ├─ Compression Ratio: 1.00x ± 0.00x
    ├─ Total Bandwidth Saved: 0.00 Mb
    ├─ Avg Inference Time: 16.466s ± 0.000s
    └─ Total Samples Evaluated: 1

  🏆 OVERALL SYSTEM SCORE: 0.5347
    ├─ Answer Quality (50%): 0.3361
    ├─ Compression Preservation (30%): 1.0000
    └─ Efficiency (20%): 0.3333

  Overall Rating: ACCEPTABLE ⭐⭐⭐
  May need tuning or optimization

✓ Results saved to experiment_results\experiment_5_realworld_scenarios.csv

📊 Summary:
                     contextual_relevance  answer_completeness  compression_ratio  bandwidth_saved_mb
scenario
conversational_qa                  0.6387                  1.0                1.0                 0.0
multi_hop_reasoning                0.4501                  0.0                1.0                 0.0
```
