# Demo Scripts Analysis

## Summary

You have **12 demo scripts** across the root directory and `examples/` folder. There is **significant redundancy** - several scripts demonstrate similar concepts with overlapping functionality.

---

## Root Directory Demos (9 files)

### 1. `demo_extraction_methods.py`

**Purpose:** Compares 3 extraction methods (simple regex, QA model, summarization)  
**Status:** ⚠️ **PARTIALLY REDUNDANT**

- Tests which extraction method works best for cross-model communication
- Similar purpose to `demo_final_extraction.py` but less comprehensive
- **Recommendation:** Can be REMOVED - `demo_final_extraction.py` is better

### 2. `demo_extraction.py`

**Purpose:** Step-by-step visual breakdown of regex-based entity extraction  
**Status:** ✅ **KEEP - EDUCATIONAL VALUE**

- Shows detailed regex patterns for numbers, names, phrases
- Explains the rule-based extraction process
- Good for understanding internals
- **Recommendation:** KEEP as educational reference

### 3. `demo_final_extraction.py`

**Purpose:** Complete comparison of ALL extraction methods (5 methods)  
**Status:** ✅ **KEEP - MOST COMPREHENSIVE**

- Compares: QA Model, Attention, YAKE, Attention (queryless), Simple regex
- Covers both query-dependent and query-independent scenarios
- Provides final recommendations (YAKE as default)
- **Recommendation:** KEEP - this is the authoritative extraction comparison

### 4. `demo_hybrid_mode.py`

**Purpose:** Compares Baseline vs Pure KV vs Hybrid (KV + Text) modes  
**Status:** ✅ **KEEP - CORE CONCEPT**

- Demonstrates why hybrid mode (KV + extracted facts) works better
- Shows the fundamental problem pure KV has with specific facts
- Tests 5 cases where pure KV fails
- **Recommendation:** KEEP - explains core Q-KVComm innovation

### 5. `demo_kv_injection.py`

**Purpose:** Shows when KV injection works and when it fails  
**Status:** ✅ **KEEP - DEMONSTRATES LIMITATIONS**

- 5 examples showing success vs failure cases
- Explains lossy nature of KV transfer
- Educational about KV cache limitations
- **Recommendation:** KEEP - important for understanding system behavior

### 6. `demo_qa_comprehensive.py`

**Purpose:** Tests QA-based extraction on longer realistic contexts  
**Status:** ⚠️ **REDUNDANT**

- Similar to other QA demos but with longer passages
- Overlaps with `demo_qa_model_comparison.py`
- **Recommendation:** Can be REMOVED or MERGED with qa_model_comparison

### 7. `demo_qa_model_comparison.py`

**Purpose:** Compares 3 different QA models (TinyRoBERTa, DistilBERT, RoBERTa)  
**Status:** ✅ **KEEP - SPECIFIC USE CASE**

- Tests model size/speed/accuracy tradeoffs
- Recommends TinyRoBERTa as best balance
- Useful if users want to choose QA model
- **Recommendation:** KEEP - provides model selection guidance

### 8. `demo_queryless_extraction.py`

**Purpose:** Demonstrates query-independent extraction (no query needed)  
**Status:** ⚠️ **PARTIALLY REDUNDANT**

- Tests attention_queryless and YAKE methods
- Similar to scenarios in `demo_final_extraction.py`
- **Recommendation:** Can be REMOVED - covered by demo_final_extraction.py

### 9. `demo_yake_complete.py`

**Purpose:** Complete demo of YAKE-based system for real agent communication  
**Status:** ✅ **KEEP - RECOMMENDED SETUP**

- Shows the recommended Q-KVComm configuration
- Full system demo with YAKE extraction
- Multiple realistic scenarios (API docs, product specs, meetings)
- **Recommendation:** KEEP - this is the "getting started" demo

---

## Examples Directory (3 files)

### 10. `examples/demo.py`

**Purpose:** Simple demonstration of Q-KVComm system  
**Status:** ✅ **KEEP - MINIMAL EXAMPLE**

- Uses Qwen2.5-1.5B model
- Shows basic setup and usage
- Good minimal starting point
- **Recommendation:** KEEP - simplest introduction

### 11. `examples/compare_performance.py`

**Purpose:** Performance comparison between Q-KVComm and traditional NL communication  
**Status:** ✅ **KEEP - BENCHMARKING**

- Compares traditional text vs Q-KVComm
- Measures speed, compression, accuracy
- Creates performance visualizations
- **Recommendation:** KEEP - important for evaluation

### 12. `examples/demo_different_models.py`

**Purpose:** Cross-model KV transfer demo (Qwen → TinyLlama)  
**Status:** ✅ **KEEP - CROSS-MODEL USE CASE**

- Shows different sender/receiver models
- Demonstrates cross-architecture transfer
- Tests layer compatibility
- **Recommendation:** KEEP - proves cross-model capability

---

## Recommendations

### ✅ KEEP (7 scripts)

1. `demo_extraction.py` - Educational breakdown of regex extraction
2. `demo_final_extraction.py` - **BEST** extraction method comparison
3. `demo_hybrid_mode.py` - Core concept: why hybrid works
4. `demo_kv_injection.py` - KV cache behavior and limitations
5. `demo_qa_model_comparison.py` - QA model selection guide
6. `demo_yake_complete.py` - **RECOMMENDED** complete setup
7. `examples/demo.py` - Minimal getting started example
8. `examples/compare_performance.py` - Performance benchmarking
9. `examples/demo_different_models.py` - Cross-model proof

### ❌ REMOVE (3 scripts)

1. `demo_extraction_methods.py` - Superseded by demo_final_extraction.py
2. `demo_qa_comprehensive.py` - Overlaps with qa_model_comparison
3. `demo_queryless_extraction.py` - Covered in demo_final_extraction.py

---

## Suggested Organization

### For New Users

1. Start with `examples/demo.py` - minimal setup
2. Read `demo_yake_complete.py` - recommended configuration
3. Explore `demo_hybrid_mode.py` - understand why hybrid works

### For Understanding Internals

4. Read `demo_extraction.py` - how extraction works
5. Read `demo_kv_injection.py` - KV cache behavior

### For Advanced Use Cases

6. Check `demo_final_extraction.py` - choose extraction method
7. Check `demo_qa_model_comparison.py` - if using QA extraction
8. Try `examples/demo_different_models.py` - cross-model transfer
9. Run `examples/compare_performance.py` - benchmark system

---

## Impact of Removal

Removing the 3 redundant scripts will:

- ✅ Reduce confusion (fewer files to navigate)
- ✅ Remove 25% of demo files
- ✅ No functionality loss (all features covered elsewhere)
- ✅ Clearer "recommended path" for new users

The remaining 9 scripts provide complete coverage of all features.
