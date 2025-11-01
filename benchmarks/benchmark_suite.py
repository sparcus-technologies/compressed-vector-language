"""Comprehensive benchmarking suite using real open-source datasets with improved agentic communication metrics"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("=" * 80)
    print("⚠️  WARNING: sentence-transformers not available")
    print("=" * 80)
    print(
        "Semantic metrics (contextual_relevance, semantic_fidelity) will use fallback methods."
    )
    print("For full semantic analysis, install with:")
    print("  pip install sentence-transformers scikit-learn")
    print("=" * 80 + "\n")


class AgenticCommunicationMetrics:
    """Metrics specifically designed for evaluating agentic communication quality"""

    def __init__(self, qkvcomm_system=None):
        """Initialize semantic similarity model if available"""
        self.semantic_model = None
        self.qkvcomm_system = qkvcomm_system
        if SEMANTIC_AVAILABLE:
            try:
                print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Semantic model loaded successfully\n")
            except Exception as e:
                print(f"✗ Could not load semantic model: {e}")
                print("  Falling back to token-based similarity\n")
                self.semantic_model = None

    def _token_based_similarity(self, text1: str, text2: str) -> float:
        """Fallback token-based similarity when semantic model unavailable"""
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def contextual_relevance(
        self, prediction: str, context: str, question: str = ""
    ) -> float:
        """
        Measure how well the prediction maintains contextual relevance
        Key for agent-to-agent communication where context transfer is critical

        NOTE: If question is provided, measures question relevance instead,
        which is more appropriate for QA tasks
        """
        # For QA tasks, measure question relevance instead of context relevance
        if question:
            if not question or not prediction:
                return 0.0

            if self.semantic_model is not None:
                try:
                    # Semantic embedding similarity with question
                    pred_emb = self.semantic_model.encode([prediction])
                    q_emb = self.semantic_model.encode([question])

                    similarity = cosine_similarity(pred_emb, q_emb)[0][0]
                    return float(max(0.0, min(1.0, similarity)))
                except Exception as e:
                    print(f"Warning: Semantic similarity failed, using fallback: {e}")
                    return self._token_based_similarity(prediction, question)
            else:
                # Fallback: token-based similarity
                return self._token_based_similarity(prediction, question)

        # Original context-based relevance (for non-QA tasks)
        if not context or not prediction:
            return 0.0

        if self.semantic_model is not None:
            try:
                # Semantic embedding similarity
                pred_emb = self.semantic_model.encode([prediction])
                ctx_emb = self.semantic_model.encode([context[:512]])  # First 512 chars

                similarity = cosine_similarity(pred_emb, ctx_emb)[0][0]
                return float(max(0.0, min(1.0, similarity)))
            except Exception as e:
                print(f"Warning: Semantic similarity failed, using fallback: {e}")
                return self._token_based_similarity(prediction, context[:512])
        else:
            # Fallback: token-based similarity
            return self._token_based_similarity(prediction, context[:512])

    def answer_completeness(self, prediction: str, ground_truth: str) -> float:
        """
        Measure if the prediction contains the essential information
        More lenient than exact match, suitable for agentic communication
        """
        pred_lower = prediction.lower().strip()
        truth_lower = ground_truth.lower().strip()

        if not truth_lower or not pred_lower:
            return 0.0

        # Exact substring match (best case)
        if truth_lower in pred_lower:
            return 1.0

        # Token overlap analysis
        truth_tokens = set(truth_lower.split())
        pred_tokens = set(pred_lower.split())

        if not truth_tokens:
            return 0.0

        # Percentage of ground truth tokens present in prediction
        overlap = len(truth_tokens & pred_tokens) / len(truth_tokens)
        return float(overlap)

    def semantic_fidelity(
        self, prediction: str, ground_truth: str, is_answerable: bool = True
    ) -> float:
        """
        Measure semantic similarity between prediction and ground truth
        Critical for evaluating if compressed KV cache preserves meaning

        Special handling for unanswerable questions to avoid misleading 0.0 scores
        """
        if not prediction or not ground_truth:
            return 0.0

        # Special handling for unanswerable questions
        if not is_answerable or ground_truth.lower().strip() == "unanswerable":
            # Check if model correctly indicates it can't answer
            unanswerable_indicators = [
                "cannot",
                "can't",
                "unable",
                "unanswerable",
                "not possible",
                "insufficient",
                "no information",
                "does not provide",
                "doesn't provide",
                "not mentioned",
                "not specified",
                "unclear",
                "not clear",
            ]
            pred_lower = prediction.lower()

            # If model indicates uncertainty/inability, score it high
            if any(indicator in pred_lower for indicator in unanswerable_indicators):
                return 0.8  # Good - recognized as unanswerable
            else:
                # Model gave a confident answer when it shouldn't have
                # But we'll use semantic similarity in case answer was reasonable
                if self.semantic_model is not None:
                    try:
                        pred_emb = self.semantic_model.encode([prediction])
                        # Use a neutral "cannot answer" phrase for comparison
                        neutral_emb = self.semantic_model.encode(
                            [
                                "This question cannot be answered based on the given information."
                            ]
                        )
                        similarity = cosine_similarity(pred_emb, neutral_emb)[0][0]
                        return float(
                            max(0.0, min(0.5, similarity))
                        )  # Cap at 0.5 since it didn't explicitly say unanswerable
                    except Exception as e:
                        return 0.0
                else:
                    return 0.0

        # Normal semantic comparison for answerable questions
        if self.semantic_model is not None:
            try:
                pred_emb = self.semantic_model.encode([prediction])
                truth_emb = self.semantic_model.encode([ground_truth])

                similarity = cosine_similarity(pred_emb, truth_emb)[0][0]
                return float(max(0.0, min(1.0, similarity)))
            except Exception as e:
                print(f"Warning: Semantic fidelity failed, using fallback: {e}")
                return self._token_based_similarity(prediction, ground_truth)
        else:
            # Fallback: enhanced token similarity with bi-gram overlap
            return self._enhanced_token_similarity(prediction, ground_truth)

    def _enhanced_token_similarity(self, text1: str, text2: str) -> float:
        """Enhanced token-based similarity with bi-grams"""
        if not text1 or not text2:
            return 0.0

        # Unigram similarity
        unigram_sim = self._token_based_similarity(text1, text2)

        # Bigram similarity
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        bigrams1 = set(zip(words1[:-1], words1[1:]))
        bigrams2 = set(zip(words2[:-1], words2[1:]))

        if bigrams1 and bigrams2:
            bigram_sim = len(bigrams1 & bigrams2) / max(len(bigrams1), len(bigrams2))
        else:
            bigram_sim = 0.0

        # Weighted combination
        return 0.7 * unigram_sim + 0.3 * bigram_sim

    def response_coherence(self, prediction: str) -> float:
        """
        Measure coherence of the generated response
        Important for multi-hop reasoning preserved through KV compression
        """
        if not prediction or len(prediction.strip()) < 3:
            return 0.0

        score = 0.0

        # Check for reasonable length (3-100 words optimal)
        word_count = len(prediction.split())
        if 3 <= word_count <= 100:
            score += 0.3
        elif 100 < word_count <= 150:
            score += 0.2
        elif word_count > 150:
            score += 0.1

        # Check for proper capitalization
        if prediction[0].isupper():
            score += 0.2

        # Check for sentence structure (ending punctuation)
        if any(punct in prediction for punct in ".!?"):
            score += 0.2

        # Check for repetition (penalize excessive repetition)
        words = prediction.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += 0.3 * unique_ratio

        return min(1.0, score)

    def communication_efficiency(
        self, prediction: str, ground_truth: str, bits_transmitted: int
    ) -> float:
        """
        Measure communication efficiency: useful information per transmitted bit
        Critical for evaluating compression effectiveness in agent communication

        Formula: efficiency = answer_quality * (1 / bits_transmitted_normalized)
        where answer_quality combines correctness and completeness
        """
        if not prediction or not ground_truth or bits_transmitted <= 0:
            return 0.0

        # Calculate answer quality (combination of completeness and correctness)
        completeness = self.answer_completeness(prediction, ground_truth)

        # Normalize bits transmitted (assume 1M bits as baseline)
        bits_baseline = 1_000_000  # 1 megabit baseline
        bits_ratio = min(1.0, bits_transmitted / bits_baseline)

        # Efficiency: higher quality with fewer bits = better
        efficiency = completeness / max(bits_ratio, 0.01)  # Avoid division by zero

        return float(min(1.0, efficiency))

    def compression_quality_score(
        self, prediction_compressed: str, prediction_baseline: str, ground_truth: str
    ) -> float:
        """
        Measure how well compression preserves answer quality
        Compares compressed vs uncompressed performance

        Score = 1.0 means no quality loss
        Score < 1.0 means quality degradation
        Score > 1.0 theoretically possible if compression helps (rare)
        """
        if not prediction_compressed or not prediction_baseline:
            return 0.0

        # Quality of compressed answer
        compressed_quality = self.answer_completeness(
            prediction_compressed, ground_truth
        )

        # Quality of baseline (uncompressed) answer
        baseline_quality = self.answer_completeness(prediction_baseline, ground_truth)

        if baseline_quality == 0:
            return 1.0 if compressed_quality == 0 else 0.0

        # Ratio: 1.0 = perfect preservation, <1.0 = degradation
        preservation_ratio = compressed_quality / baseline_quality

        return float(preservation_ratio)

    def semantic_preservation(
        self, prediction_compressed: str, prediction_baseline: str
    ) -> float:
        """
        Measure semantic similarity between compressed and baseline outputs
        This directly evaluates if KV compression preserves meaning

        KEY METRIC for compression systems: Are we preserving semantic intent?
        """
        if not prediction_compressed or not prediction_baseline:
            return 0.0

        if self.semantic_model is not None:
            try:
                compressed_emb = self.semantic_model.encode([prediction_compressed])
                baseline_emb = self.semantic_model.encode([prediction_baseline])

                similarity = cosine_similarity(compressed_emb, baseline_emb)[0][0]
                return float(max(0.0, min(1.0, similarity)))
            except Exception as e:
                print(f"Warning: Semantic preservation failed: {e}")
                return self._token_based_similarity(
                    prediction_compressed, prediction_baseline
                )
        else:
            return self._token_based_similarity(
                prediction_compressed, prediction_baseline
            )

    def information_throughput(
        self, prediction: str, ground_truth: str, inference_time: float
    ) -> float:
        """
        Measure information throughput: correct information per second
        Important for real-time agentic communication systems

        Formula: throughput = (answer_quality / inference_time)
        Higher is better
        """
        if inference_time <= 0 or not prediction:
            return 0.0

        quality = self.answer_completeness(prediction, ground_truth)
        throughput = quality / inference_time

        # Normalize to reasonable scale (1.0 = 1 quality unit per second)
        return float(min(10.0, throughput))  # Cap at 10 for normalization

    def compute_all_metrics(
        self,
        prediction: str,
        ground_truth: str,
        context: str = "",
        question: str = "",
        is_answerable: bool = True,
        bits_transmitted: int = 0,
        inference_time: float = 0.0,
        baseline_prediction: str = None,
    ) -> dict:
        """Compute all agentic communication metrics"""

        metrics = {
            # Original Quality Metrics
            "contextual_relevance": self.contextual_relevance(
                prediction, context, question
            ),
            "answer_completeness": self.answer_completeness(prediction, ground_truth),
            "semantic_fidelity": self.semantic_fidelity(
                prediction, ground_truth, is_answerable
            ),
            "response_coherence": self.response_coherence(prediction),
        }

        # New Compression-Specific Metrics
        if bits_transmitted > 0:
            metrics["communication_efficiency"] = self.communication_efficiency(
                prediction, ground_truth, bits_transmitted
            )

        if inference_time > 0:
            metrics["information_throughput"] = self.information_throughput(
                prediction, ground_truth, inference_time
            )

        # Compression Quality (requires baseline comparison)
        if baseline_prediction:
            metrics["compression_quality_score"] = self.compression_quality_score(
                prediction, baseline_prediction, ground_truth
            )
            metrics["semantic_preservation"] = self.semantic_preservation(
                prediction, baseline_prediction
            )

        # Add debug flag
        metrics["_using_semantic_model"] = self.semantic_model is not None

        return metrics


class BenchmarkSuite:
    """Comprehensive benchmark suite for Q-KVComm using real datasets"""

    def __init__(
        self,
        qkvcomm_system,
        output_dir: str = "benchmark_results",
        enable_baseline: bool = False,
    ):
        """
        Initialize benchmark suite

        Args:
            qkvcomm_system: Q-KVComm system to benchmark
            output_dir: Directory to save results
            enable_baseline: Whether to run baseline comparisons (slower but more insightful)
        """
        self.qkvcomm = qkvcomm_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)
        self.agentic_metrics = AgenticCommunicationMetrics(qkvcomm_system)
        self.enable_baseline = enable_baseline

    def load_squad(self, max_samples: int = 100) -> List[Dict]:
        """Load SQuAD 2.0 dataset - extractive QA"""
        print("Loading SQuAD 2.0 dataset...")
        dataset = load_dataset("squad_v2", split="validation")

        samples = []

        for item in dataset:
            if len(samples) >= max_samples:
                break

            # Include both answerable and unanswerable
            if item["answers"]["text"]:
                answer = item["answers"]["text"][0]
                is_answerable = True
            else:
                answer = "unanswerable"
                is_answerable = False

            samples.append(
                {
                    "id": item["id"],
                    "context": item["context"],
                    "question": item["question"],
                    "answers": answer,
                    "is_answerable": is_answerable,
                    "dataset": "squad",
                }
            )

        print(
            f"Loaded {len(samples)} SQuAD samples "
            f"({sum(1 for s in samples if s['is_answerable'])} answerable, "
            f"{sum(1 for s in samples if not s['is_answerable'])} unanswerable)"
        )
        return samples

    def load_hotpot_qa(self, max_samples: int = 100) -> List[Dict]:
        """Load HotpotQA dataset - multi-hop reasoning"""
        print("Loading HotpotQA dataset...")
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        except Exception as e:
            print(f"Warning: Could not load HotpotQA: {e}")
            return []

        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 3:
                break
            if len(samples) >= max_samples:
                break

            try:
                # Concatenate supporting facts as context
                context_parts = []
                for title, sentences in zip(
                    item["context"]["title"], item["context"]["sentences"]
                ):
                    context_parts.extend(sentences)
                context = " ".join(context_parts[:800])

                if context and item["answer"]:
                    samples.append(
                        {
                            "id": item["id"],
                            "context": context,
                            "question": item["question"],
                            "answers": item["answer"],
                            "is_answerable": True,
                            "dataset": "hotpot_qa",
                        }
                    )
            except Exception:
                continue

        print(f"Loaded {len(samples)} HotpotQA samples")
        return samples

    def load_natural_questions(self, max_samples: int = 100) -> List[Dict]:
        """Load Natural Questions dataset - open domain QA"""
        print("Loading Natural Questions dataset...")
        try:
            dataset = load_dataset("natural_questions", split="validation")
        except Exception as e:
            print(f"Warning: Could not load Natural Questions: {e}")
            return []

        samples = []
        checked = 0

        for item in dataset:
            checked += 1
            if checked > max_samples * 5:  # Check more to find valid samples
                break
            if len(samples) >= max_samples:
                break

            try:
                if item["annotations"]["short_answers"]:
                    short_ans = item["annotations"]["short_answers"][0]
                    if short_ans["text"]:
                        doc_text = item["document"]["tokens"]["token"]
                        context = " ".join(doc_text[:500])

                        samples.append(
                            {
                                "id": f"nq_{checked}",
                                "context": context,
                                "question": item["question"]["text"],
                                "answers": short_ans["text"][0],
                                "is_answerable": True,
                                "dataset": "natural_questions",
                            }
                        )
            except Exception:
                continue

        print(
            f"Loaded {len(samples)} Natural Questions samples (checked {checked} items)"
        )
        return samples

    def load_coqa(self, max_samples: int = 100) -> List[Dict]:
        """Load CoQA dataset - conversational QA"""
        print("Loading CoQA dataset...")
        try:
            dataset = load_dataset("coqa", split="validation")
        except Exception as e:
            print(f"Warning: Could not load CoQA dataset: {e}")
            return []

        samples = []
        try:
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                if item["questions"] and item["answers"]["input_text"]:
                    samples.append(
                        {
                            "id": f"coqa_{i}",  # Use index instead of item['id']
                            "context": item["story"],
                            "question": item["questions"][0],
                            "answers": item["answers"]["input_text"][0],
                            "is_answerable": True,
                            "dataset": "coqa",
                        }
                    )
        except Exception as e:
            print(f"Warning: Failed to load coqa: {e}")
            return samples  # Return what we have so far

        print(f"Loaded {len(samples)} CoQA samples")
        return samples

    def load_narrativeqa(self, max_samples: int = 100) -> List[Dict]:
        """Load NarrativeQA dataset - reading comprehension"""
        print("Loading NarrativeQA dataset...")
        try:
            dataset = load_dataset("narrativeqa", split="validation")
        except Exception as e:
            print(f"Warning: Could not load NarrativeQA: {e}")
            return []

        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:
                break
            if len(samples) >= max_samples:
                break

            try:
                context = item["document"]["summary"]["text"]

                if context and item["answers"]:
                    answer = item["answers"][0]["text"] if item["answers"] else ""

                    if answer:
                        samples.append(
                            {
                                "id": f"narrativeqa_{i}",
                                "context": context[:1000],
                                "question": item["question"]["text"],
                                "answers": answer,
                                "is_answerable": True,
                                "dataset": "narrativeqa",
                            }
                        )
            except Exception:
                continue

        print(f"Loaded {len(samples)} NarrativeQA samples")
        return samples

    def evaluate_sample(self, sample: Dict, max_new_tokens: int = 50) -> Dict:
        """Evaluate single sample with agentic communication metrics"""
        try:
            # Optional: Run baseline (no compression) for comparison
            baseline_output = None
            baseline_time = 0.0

            if self.enable_baseline:
                try:
                    # Temporarily disable compression
                    original_mode = self.qkvcomm.config.mode
                    self.qkvcomm.config.mode = "baseline"
                    self.qkvcomm.config.quantization_enabled = False
                    self.qkvcomm.config.calibration_enabled = False

                    baseline_start = time.time()
                    baseline_output, _ = self.qkvcomm.communicate(
                        context=sample["context"],
                        query=sample["question"],
                        max_new_tokens=max_new_tokens,
                    )
                    baseline_time = time.time() - baseline_start

                    # Restore compression settings
                    self.qkvcomm.config.mode = original_mode
                    if original_mode == "full":
                        self.qkvcomm.config.quantization_enabled = True
                        self.qkvcomm.config.calibration_enabled = True
                    elif original_mode == "quantization_only":
                        self.qkvcomm.config.quantization_enabled = True
                except Exception as e:
                    print(f"Warning: Baseline evaluation failed: {e}")
                    baseline_output = None

            start_time = time.time()

            # Run Q-KVComm with compression
            output, comm_metrics = self.qkvcomm.communicate(
                context=sample["context"],
                query=sample["question"],
                max_new_tokens=max_new_tokens,
            )

            inference_time = time.time() - start_time

            # Compute agentic communication metrics with enhanced parameters
            agentic = self.agentic_metrics.compute_all_metrics(
                prediction=output,
                ground_truth=sample["answers"],
                context=sample["context"],
                question=sample["question"],
                is_answerable=sample.get("is_answerable", True),
                bits_transmitted=comm_metrics.get("total_bits_compressed", 0),
                inference_time=inference_time,
                baseline_prediction=baseline_output,
            )

            result = {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "question": sample["question"],
                "prediction": output,
                "ground_truth": sample["answers"],
                "is_answerable": sample.get("is_answerable", True),
                # Core Quality Metrics
                "contextual_relevance": agentic["contextual_relevance"],
                "answer_completeness": agentic["answer_completeness"],
                "semantic_fidelity": agentic["semantic_fidelity"],
                "response_coherence": agentic["response_coherence"],
                # New Compression-Specific Metrics
                "communication_efficiency": agentic.get(
                    "communication_efficiency", 0.0
                ),
                "information_throughput": agentic.get("information_throughput", 0.0),
                "compression_quality_score": agentic.get(
                    "compression_quality_score", None
                ),
                "semantic_preservation": agentic.get("semantic_preservation", None),
                # Communication Metrics
                "inference_time": inference_time,
                "compression_ratio": comm_metrics.get("avg_compression_ratio", 1.0),
                "layers_transmitted": comm_metrics.get("num_layers_transmitted", 0),
                "bits_original": comm_metrics.get("total_bits_original", 0),
                "bits_compressed": comm_metrics.get("total_bits_compressed", 0),
                "bits_saved": comm_metrics.get("total_bits_original", 0)
                - comm_metrics.get("total_bits_compressed", 0),
                # Baseline comparison (if available)
                "baseline_prediction": (
                    baseline_output if self.enable_baseline else None
                ),
                "baseline_time": baseline_time if self.enable_baseline else None,
                # Metadata
                "using_semantic_model": agentic.get("_using_semantic_model", False),
                "success": True,
            }

        except Exception as e:
            result = {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "success": False,
                "error": str(e),
            }
            print(f"Error evaluating {sample['id']}: {e}")

        return result

    def run_benchmark(
        self, dataset_names: List[str], max_samples: int = 100, max_new_tokens: int = 50
    ):
        """Run comprehensive benchmark"""
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not installed. Run: pip install datasets")
            return

        print("=" * 80)
        print("Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK")
        print("=" * 80)
        print(f"\nDatasets: {dataset_names}")
        print(f"Max samples per dataset: {max_samples}")
        print(f"Max new tokens: {max_new_tokens}")
        print(
            f"Semantic model: {'✓ Loaded' if self.agentic_metrics.semantic_model else '✗ Using fallback'}\n"
        )

        # Load datasets
        all_samples = []
        dataset_loaders = {
            "squad": self.load_squad,
            "hotpot_qa": self.load_hotpot_qa,
            "natural_questions": self.load_natural_questions,
            "coqa": self.load_coqa,
            "narrativeqa": self.load_narrativeqa,
        }

        for dataset_name in dataset_names:
            if dataset_name in dataset_loaders:
                try:
                    samples = dataset_loaders[dataset_name](max_samples)
                    all_samples.extend(samples)
                except Exception as e:
                    print(f"Warning: Failed to load {dataset_name}: {e}")
            else:
                print(f"Warning: Unknown dataset {dataset_name}")

        if not all_samples:
            print("Error: No samples loaded!")
            return

        print(f"\nTotal samples loaded: {len(all_samples)}")

        # Calibration
        calibration_contexts = [
            s["context"] for s in all_samples[: min(50, len(all_samples))]
        ]
        print(f"Using {len(calibration_contexts)} samples for calibration...")
        self.qkvcomm.calibrate(calibration_contexts)

        # Evaluation
        print(f"\nEvaluating {len(all_samples)} samples...\n")

        for sample in tqdm(all_samples, desc="Evaluating"):
            result = self.evaluate_sample(sample, max_new_tokens)
            self.results[sample["dataset"]].append(result)

        # Save and analyze
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(dict(self.results), f, indent=2)

        # CSV
        all_results = []
        for dataset, results in self.results.items():
            all_results.extend(results)

        df = pd.DataFrame(all_results)
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        print(f"\nResults saved to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")

    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("AGENTIC COMMUNICATION BENCHMARK SUMMARY")
        print("=" * 80 + "\n")

        overall_metrics = defaultdict(list)

        for dataset_name, results in self.results.items():
            successful = [r for r in results if r.get("success", False)]

            if not successful:
                continue

            print(f"\n{dataset_name.upper()}")
            print("-" * 60)
            print(
                f"  Total: {len(results)} | Successful: {len(successful)} | Success rate: {len(successful)/len(results)*100:.1f}%"
            )

            # Core Quality metrics
            avg_context_rel = np.mean([r["contextual_relevance"] for r in successful])
            avg_completeness = np.mean([r["answer_completeness"] for r in successful])
            avg_fidelity = np.mean([r["semantic_fidelity"] for r in successful])
            avg_coherence = np.mean([r["response_coherence"] for r in successful])

            # New Compression-Specific metrics
            avg_comm_eff = np.mean(
                [r.get("communication_efficiency", 0) for r in successful]
            )
            avg_throughput = np.mean(
                [r.get("information_throughput", 0) for r in successful]
            )

            # Baseline comparison metrics (if available)
            compression_quality_scores = [
                r.get("compression_quality_score")
                for r in successful
                if r.get("compression_quality_score") is not None
            ]
            semantic_preservation_scores = [
                r.get("semantic_preservation")
                for r in successful
                if r.get("semantic_preservation") is not None
            ]

            print(f"\n  📊 Core Quality Metrics:")
            print(
                f"    ├─ Contextual Relevance: {avg_context_rel:.4f} {'⭐⭐⭐' if avg_context_rel > 0.7 else '⭐⭐' if avg_context_rel > 0.5 else '⭐'}"
            )
            print(
                f"    ├─ Answer Completeness: {avg_completeness:.4f} {'⭐⭐⭐' if avg_completeness > 0.7 else '⭐⭐' if avg_completeness > 0.5 else '⭐'}"
            )
            print(
                f"    ├─ Semantic Fidelity: {avg_fidelity:.4f} {'⭐⭐⭐' if avg_fidelity > 0.7 else '⭐⭐' if avg_fidelity > 0.5 else '⭐'}"
            )
            print(
                f"    └─ Response Coherence: {avg_coherence:.4f} {'⭐⭐⭐' if avg_coherence > 0.7 else '⭐⭐' if avg_coherence > 0.5 else '⭐'}"
            )

            # New compression-specific metrics
            print(f"\n  🔬 Compression-Specific Metrics:")
            print(
                f"    ├─ Communication Efficiency: {avg_comm_eff:.4f} {'⭐⭐⭐' if avg_comm_eff > 0.7 else '⭐⭐' if avg_comm_eff > 0.5 else '⭐'}"
            )
            print(
                f"    └─ Information Throughput: {avg_throughput:.4f} units/s {'⭐⭐⭐' if avg_throughput > 0.5 else '⭐⭐' if avg_throughput > 0.2 else '⭐'}"
            )

            # Baseline comparison (if enabled)
            if compression_quality_scores:
                avg_comp_quality = np.mean(compression_quality_scores)
                avg_sem_preservation = (
                    np.mean(semantic_preservation_scores)
                    if semantic_preservation_scores
                    else 0
                )
                print(f"\n  🎯 Compression Quality (vs Baseline):")
                print(
                    f"    ├─ Quality Preservation: {avg_comp_quality:.4f} (1.0 = no loss) {'✓' if avg_comp_quality > 0.9 else '⚠' if avg_comp_quality > 0.7 else '✗'}"
                )
                print(
                    f"    └─ Semantic Preservation: {avg_sem_preservation:.4f} {'✓' if avg_sem_preservation > 0.9 else '⚠' if avg_sem_preservation > 0.7 else '✗'}"
                )

            # System efficiency metrics
            avg_time = np.mean([r["inference_time"] for r in successful])
            avg_comp = np.mean([r["compression_ratio"] for r in successful])
            avg_layers = np.mean([r["layers_transmitted"] for r in successful])
            total_bits_saved = sum([r["bits_saved"] for r in successful])

            print(f"\n  🚀 System Performance:")
            print(
                f"    ├─ Compression Ratio: {avg_comp:.2f}x {'⭐⭐⭐' if avg_comp > 2.0 else '⭐⭐' if avg_comp > 1.5 else '⭐'}"
            )
            print(f"    ├─ Bandwidth Saved: {total_bits_saved/1e6:.2f} Mb")
            print(f"    ├─ Avg Layers Transmitted: {avg_layers:.1f}")
            print(f"    └─ Avg Inference Time: {avg_time:.3f}s")

            # Collect overall
            for key in [
                "contextual_relevance",
                "answer_completeness",
                "semantic_fidelity",
                "response_coherence",
                "inference_time",
                "compression_ratio",
                "layers_transmitted",
                "bits_saved",
            ]:
                overall_metrics[key].extend([r[key] for r in successful])

            # Collect new metrics (may not exist in all results)
            for key in ["communication_efficiency", "information_throughput"]:
                overall_metrics[key].extend(
                    [r.get(key, 0) for r in successful if r.get(key) is not None]
                )

            # Collect baseline comparison metrics if available
            for key in ["compression_quality_score", "semantic_preservation"]:
                overall_metrics[key].extend(
                    [r.get(key) for r in successful if r.get(key) is not None]
                )

        # Overall summary
        if overall_metrics["answer_completeness"]:
            print("\n" + "=" * 80)
            print("OVERALL PERFORMANCE")
            print("=" * 80)

            print(f"\n  🎯 CORE QUALITY METRICS:")
            print(
                f"    ├─ Contextual Relevance: {np.mean(overall_metrics['contextual_relevance']):.4f} ± {np.std(overall_metrics['contextual_relevance']):.4f}"
            )
            print(
                f"    ├─ Answer Completeness: {np.mean(overall_metrics['answer_completeness']):.4f} ± {np.std(overall_metrics['answer_completeness']):.4f}"
            )
            print(
                f"    ├─ Semantic Fidelity: {np.mean(overall_metrics['semantic_fidelity']):.4f} ± {np.std(overall_metrics['semantic_fidelity']):.4f}"
            )
            print(
                f"    └─ Response Coherence: {np.mean(overall_metrics['response_coherence']):.4f} ± {np.std(overall_metrics['response_coherence']):.4f}"
            )

            # New compression-specific metrics
            if overall_metrics.get("communication_efficiency"):
                print(f"\n  🔬 COMPRESSION-SPECIFIC METRICS:")
                print(
                    f"    ├─ Communication Efficiency: {np.mean(overall_metrics['communication_efficiency']):.4f} ± {np.std(overall_metrics['communication_efficiency']):.4f}"
                )
                print(
                    f"    └─ Information Throughput: {np.mean(overall_metrics['information_throughput']):.4f} ± {np.std(overall_metrics['information_throughput']):.4f} units/s"
                )

            # Baseline comparison summary
            if overall_metrics.get("compression_quality_score"):
                print(f"\n  🎯 COMPRESSION QUALITY (vs Baseline):")
                print(
                    f"    ├─ Quality Preservation: {np.mean(overall_metrics['compression_quality_score']):.4f} ± {np.std(overall_metrics['compression_quality_score']):.4f}"
                )
                if overall_metrics.get("semantic_preservation"):
                    print(
                        f"    └─ Semantic Preservation: {np.mean(overall_metrics['semantic_preservation']):.4f} ± {np.std(overall_metrics['semantic_preservation']):.4f}"
                    )

            print(f"\n  ⚡ SYSTEM PERFORMANCE:")
            print(
                f"    ├─ Compression Ratio: {np.mean(overall_metrics['compression_ratio']):.2f}x ± {np.std(overall_metrics['compression_ratio']):.2f}x"
            )
            print(
                f"    ├─ Total Bandwidth Saved: {sum(overall_metrics['bits_saved'])/1e6:.2f} Mb"
            )
            print(
                f"    ├─ Avg Inference Time: {np.mean(overall_metrics['inference_time']):.3f}s ± {np.std(overall_metrics['inference_time']):.3f}s"
            )
            print(
                f"    └─ Total Samples Evaluated: {len(overall_metrics['answer_completeness'])}"
            )

            # Composite scores
            quality_score = np.mean(
                [
                    np.mean(overall_metrics["contextual_relevance"]),
                    np.mean(overall_metrics["answer_completeness"]),
                    np.mean(overall_metrics["response_coherence"]),
                    np.mean(overall_metrics["semantic_fidelity"]),
                ]
            )

            compression_ratio_mean = np.mean(overall_metrics["compression_ratio"])
            efficiency_score = min(1.0, compression_ratio_mean / 3.0)

            # Overall system score combines quality and efficiency
            if overall_metrics.get("compression_quality_score"):
                # If we have baseline comparison, use preservation ratio
                preservation = np.mean(overall_metrics["compression_quality_score"])
                system_score = (
                    (quality_score * 0.5)
                    + (preservation * 0.3)
                    + (efficiency_score * 0.2)
                )

                print(f"\n  🏆 OVERALL SYSTEM SCORE: {system_score:.4f}")
                print(f"    ├─ Answer Quality (50%): {quality_score:.4f}")
                print(f"    ├─ Compression Preservation (30%): {preservation:.4f}")
                print(f"    └─ Efficiency (20%): {efficiency_score:.4f}")
            else:
                # Without baseline, weight quality more heavily
                system_score = (quality_score * 0.7) + (efficiency_score * 0.3)

                print(f"\n  🏆 OVERALL SYSTEM SCORE: {system_score:.4f}")
                print(f"    ├─ Answer Quality (70%): {quality_score:.4f}")
                print(f"    └─ Efficiency (30%): {efficiency_score:.4f}")

            # Add interpretation
            if system_score > 0.7:
                rating = "EXCELLENT ⭐⭐⭐⭐⭐"
                desc = "Strong agentic communication with effective compression"
            elif system_score > 0.6:
                rating = "GOOD ⭐⭐⭐⭐"
                desc = "Acceptable performance for most applications"
            elif system_score > 0.5:
                rating = "ACCEPTABLE ⭐⭐⭐"
                desc = "May need tuning or optimization"
            else:
                rating = "NEEDS IMPROVEMENT ⭐⭐"
                desc = "Significant improvements needed"

            print(f"\n  Overall Rating: {rating}")
            print(f"  {desc}")
