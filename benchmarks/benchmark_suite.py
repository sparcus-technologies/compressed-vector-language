"""Comprehensive benchmarking suite using real open-source datasets with improved agentic communication metrics"""

import torch
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import pandas as pd

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
    print("Semantic metrics (contextual_relevance, semantic_fidelity) will use fallback methods.")
    print("For full semantic analysis, install with:")
    print("  pip install sentence-transformers scikit-learn")
    print("=" * 80 + "\n")


class AgenticCommunicationMetrics:
    """Metrics specifically designed for evaluating agentic communication quality"""
    
    def __init__(self):
        """Initialize semantic similarity model if available"""
        self.semantic_model = None
        if SEMANTIC_AVAILABLE:
            try:
                print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
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
    
    def contextual_relevance(self, prediction: str, context: str) -> float:
        """
        Measure how well the prediction maintains contextual relevance
        Key for agent-to-agent communication where context transfer is critical
        """
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
    
    def semantic_fidelity(self, prediction: str, ground_truth: str) -> float:
        """
        Measure semantic similarity between prediction and ground truth
        Critical for evaluating if compressed KV cache preserves meaning
        """
        if not prediction or not ground_truth:
            return 0.0
        
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
        if any(punct in prediction for punct in '.!?'):
            score += 0.2
        
        # Check for repetition (penalize excessive repetition)
        words = prediction.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score += 0.3 * unique_ratio
        
        return min(1.0, score)
    
    def information_density(self, prediction: str, ground_truth: str) -> float:
        """
        Measure information density: how much relevant info per token
        Critical for evaluating compression efficiency in agent communication
        """
        if not prediction or not ground_truth:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        truth_tokens = set(ground_truth.lower().split())
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Count how many prediction tokens are relevant (appear in ground truth)
        relevant_count = sum(1 for token in pred_tokens if token in truth_tokens)
        
        # Density = relevant tokens / total tokens
        # Penalize overly long responses
        density = relevant_count / len(pred_tokens)
        
        # Bonus for conciseness (if length is reasonable)
        if len(pred_tokens) <= 30:
            density *= 1.1  # 10% bonus for concise answers
        
        return float(min(1.0, density))
    
    def compute_all_metrics(
        self, 
        prediction: str, 
        ground_truth: str,
        context: str = ""
    ) -> dict:
        """Compute all agentic communication metrics"""
        
        metrics = {
            'contextual_relevance': self.contextual_relevance(prediction, context),
            'answer_completeness': self.answer_completeness(prediction, ground_truth),
            'semantic_fidelity': self.semantic_fidelity(prediction, ground_truth),
            'response_coherence': self.response_coherence(prediction),
            'information_density': self.information_density(prediction, ground_truth),
        }
        
        # Add debug flag
        metrics['_using_semantic_model'] = self.semantic_model is not None
        
        return metrics


class BenchmarkSuite:
    """Comprehensive benchmark suite for Q-KVComm using real datasets"""
    
    def __init__(self, qkvcomm_system, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite
        
        Args:
            qkvcomm_system: Q-KVComm system to benchmark
            output_dir: Directory to save results
        """
        self.qkvcomm = qkvcomm_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(list)
        self.agentic_metrics = AgenticCommunicationMetrics()
        
    def load_squad(self, max_samples: int = 100) -> List[Dict]:
        """Load SQuAD 2.0 dataset - extractive QA"""
        print("Loading SQuAD 2.0 dataset...")
        dataset = load_dataset("squad_v2", split="validation")
        
        samples = []
        
        for item in dataset:
            if len(samples) >= max_samples:
                break
            
            # Include both answerable and unanswerable
            if item['answers']['text']:
                answer = item['answers']['text'][0]
                is_answerable = True
            else:
                answer = "unanswerable"
                is_answerable = False
            
            samples.append({
                'id': item['id'],
                'context': item['context'],
                'question': item['question'],
                'answers': answer,
                'is_answerable': is_answerable,
                'dataset': 'squad',
            })
        
        print(f"Loaded {len(samples)} SQuAD samples "
              f"({sum(1 for s in samples if s['is_answerable'])} answerable, "
              f"{sum(1 for s in samples if not s['is_answerable'])} unanswerable)")
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
                for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                    context_parts.extend(sentences)
                context = ' '.join(context_parts[:800])
                
                if context and item['answer']:
                    samples.append({
                        'id': item['id'],
                        'context': context,
                        'question': item['question'],
                        'answers': item['answer'],
                        'is_answerable': True,
                        'dataset': 'hotpot_qa',
                    })
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
                if item['annotations']['short_answers']:
                    short_ans = item['annotations']['short_answers'][0]
                    if short_ans['text']:
                        doc_text = item['document']['tokens']['token']
                        context = ' '.join(doc_text[:500])
                        
                        samples.append({
                            'id': f"nq_{checked}",
                            'context': context,
                            'question': item['question']['text'],
                            'answers': short_ans['text'][0],
                            'is_answerable': True,
                            'dataset': 'natural_questions',
                        })
            except Exception:
                continue
        
        print(f"Loaded {len(samples)} Natural Questions samples (checked {checked} items)")
        return samples
    
    def load_coqa(self, max_samples: int = 100) -> List[Dict]:
        """Load CoQA dataset - conversational QA"""
        print("Loading CoQA dataset...")
        try:
            dataset = load_dataset("coqa", split="validation")
        except Exception as e:
            print(f"Warning: Could not load CoQA: {e}")
            return []
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            
            if item['questions'] and item['answers']['input_text']:
                samples.append({
                    'id': f"coqa_{item['id']}",
                    'context': item['story'],
                    'question': item['questions'][0],
                    'answers': item['answers']['input_text'][0],
                    'is_answerable': True,
                    'dataset': 'coqa',
                })
        
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
                context = item['document']['summary']['text']
                
                if context and item['answers']:
                    answer = item['answers'][0]['text'] if item['answers'] else ""
                    
                    if answer:
                        samples.append({
                            'id': f"narrativeqa_{i}",
                            'context': context[:1000],
                            'question': item['question']['text'],
                            'answers': answer,
                            'is_answerable': True,
                            'dataset': 'narrativeqa',
                        })
            except Exception:
                continue
        
        print(f"Loaded {len(samples)} NarrativeQA samples")
        return samples
    
    def evaluate_sample(self, sample: Dict, max_new_tokens: int = 50) -> Dict:
        """Evaluate single sample with agentic communication metrics"""
        try:
            start_time = time.time()
            
            # Run Q-KVComm
            output, comm_metrics = self.qkvcomm.communicate(
                context=sample['context'],
                query=sample['question'],
                max_new_tokens=max_new_tokens
            )
            
            inference_time = time.time() - start_time
            
            # Compute agentic communication metrics
            agentic = self.agentic_metrics.compute_all_metrics(
                prediction=output,
                ground_truth=sample['answers'],
                context=sample['context']
            )
            
            result = {
                'id': sample['id'],
                'dataset': sample['dataset'],
                'question': sample['question'],
                'prediction': output,
                'ground_truth': sample['answers'],
                'is_answerable': sample.get('is_answerable', True),
                
                # Agentic Communication Quality Metrics
                'contextual_relevance': agentic['contextual_relevance'],
                'answer_completeness': agentic['answer_completeness'],
                'semantic_fidelity': agentic['semantic_fidelity'],
                'response_coherence': agentic['response_coherence'],
                'information_density': agentic['information_density'],
                
                # Communication Efficiency Metrics
                'inference_time': inference_time,
                'compression_ratio': comm_metrics.get('avg_compression_ratio', 1.0),
                'layers_transmitted': comm_metrics.get('num_layers_transmitted', 0),
                'bits_original': comm_metrics.get('total_bits_original', 0),
                'bits_compressed': comm_metrics.get('total_bits_compressed', 0),
                'bits_saved': comm_metrics.get('total_bits_original', 0) - comm_metrics.get('total_bits_compressed', 0),
                
                # Metadata
                'using_semantic_model': agentic.get('_using_semantic_model', False),
                'success': True
            }
            
        except Exception as e:
            result = {
                'id': sample['id'],
                'dataset': sample['dataset'],
                'success': False,
                'error': str(e)
            }
            print(f"Error evaluating {sample['id']}: {e}")
        
        return result
    
    def run_benchmark(self, dataset_names: List[str], max_samples: int = 100, max_new_tokens: int = 50):
        """Run comprehensive benchmark"""
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not installed. Run: pip install datasets")
            return
        
        print("="*80)
        print("Q-KVCOMM AGENTIC COMMUNICATION BENCHMARK")
        print("="*80)
        print(f"\nDatasets: {dataset_names}")
        print(f"Max samples per dataset: {max_samples}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Semantic model: {'✓ Loaded' if self.agentic_metrics.semantic_model else '✗ Using fallback'}\n")
        
        # Load datasets
        all_samples = []
        dataset_loaders = {
            'squad': self.load_squad,
            'hotpot_qa': self.load_hotpot_qa,
            'natural_questions': self.load_natural_questions,
            'coqa': self.load_coqa,
            'narrativeqa': self.load_narrativeqa,
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
        calibration_contexts = [s['context'] for s in all_samples[:min(50, len(all_samples))]]
        print(f"Using {len(calibration_contexts)} samples for calibration...")
        self.qkvcomm.calibrate(calibration_contexts)
        
        # Evaluation
        print(f"\nEvaluating {len(all_samples)} samples...\n")
        
        for sample in tqdm(all_samples, desc="Evaluating"):
            result = self.evaluate_sample(sample, max_new_tokens)
            self.results[sample['dataset']].append(result)
        
        # Save and analyze
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
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
        print("\n" + "="*80)
        print("AGENTIC COMMUNICATION BENCHMARK SUMMARY")
        print("="*80 + "\n")
        
        overall_metrics = defaultdict(list)
        
        for dataset_name, results in self.results.items():
            successful = [r for r in results if r.get('success', False)]
            
            if not successful:
                continue
            
            print(f"\n{dataset_name.upper()}")
            print("-" * 60)
            print(f"  Total: {len(results)} | Successful: {len(successful)} | Success rate: {len(successful)/len(results)*100:.1f}%")
            
            # Quality metrics
            avg_context_rel = np.mean([r['contextual_relevance'] for r in successful])
            avg_completeness = np.mean([r['answer_completeness'] for r in successful])
            avg_fidelity = np.mean([r['semantic_fidelity'] for r in successful])
            avg_coherence = np.mean([r['response_coherence'] for r in successful])
            avg_density = np.mean([r['information_density'] for r in successful])
            
            print(f"\n  📊 Agentic Communication Quality:")
            print(f"    ├─ Contextual Relevance: {avg_context_rel:.4f} {'⭐⭐⭐' if avg_context_rel > 0.7 else '⭐⭐' if avg_context_rel > 0.5 else '⭐'}")
            print(f"    ├─ Answer Completeness: {avg_completeness:.4f} {'⭐⭐⭐' if avg_completeness > 0.7 else '⭐⭐' if avg_completeness > 0.5 else '⭐'}")
            print(f"    ├─ Semantic Fidelity: {avg_fidelity:.4f} {'⭐⭐⭐' if avg_fidelity > 0.7 else '⭐⭐' if avg_fidelity > 0.5 else '⭐'}")
            print(f"    ├─ Response Coherence: {avg_coherence:.4f} {'⭐⭐⭐' if avg_coherence > 0.7 else '⭐⭐' if avg_coherence > 0.5 else '⭐'}")
            print(f"    └─ Information Density: {avg_density:.4f} {'⭐⭐⭐' if avg_density > 0.5 else '⭐⭐' if avg_density > 0.3 else '⭐'}")
            
            # Efficiency metrics
            avg_time = np.mean([r['inference_time'] for r in successful])
            avg_comp = np.mean([r['compression_ratio'] for r in successful])
            avg_layers = np.mean([r['layers_transmitted'] for r in successful])
            total_bits_saved = sum([r['bits_saved'] for r in successful])
            
            print(f"\n  🚀 Communication Efficiency:")
            print(f"    ├─ Compression: {avg_comp:.2f}x {'⭐⭐⭐' if avg_comp > 2.0 else '⭐⭐' if avg_comp > 1.5 else '⭐'}")
            print(f"    ├─ Bandwidth Saved: {total_bits_saved/1e6:.2f} Mb")
            print(f"    ├─ Avg Layers: {avg_layers:.1f}")
            print(f"    └─ Avg Latency: {avg_time:.3f}s")
            
            # Collect overall
            for key in ['contextual_relevance', 'answer_completeness', 'semantic_fidelity', 
                       'response_coherence', 'information_density', 'inference_time',
                       'compression_ratio', 'layers_transmitted', 'bits_saved']:
                overall_metrics[key].extend([r[key] for r in successful])
        
        # Overall summary
        if overall_metrics['answer_completeness']:
            print("\n" + "="*80)
            print("OVERALL PERFORMANCE")
            print("="*80)
            
            print(f"\n  🎯 QUALITY METRICS:")
            print(f"    ├─ Contextual Relevance: {np.mean(overall_metrics['contextual_relevance']):.4f} ± {np.std(overall_metrics['contextual_relevance']):.4f}")
            print(f"    ├─ Answer Completeness: {np.mean(overall_metrics['answer_completeness']):.4f} ± {np.std(overall_metrics['answer_completeness']):.4f}")
            print(f"    ├─ Semantic Fidelity: {np.mean(overall_metrics['semantic_fidelity']):.4f} ± {np.std(overall_metrics['semantic_fidelity']):.4f}")
            print(f"    ├─ Response Coherence: {np.mean(overall_metrics['response_coherence']):.4f} ± {np.std(overall_metrics['response_coherence']):.4f}")
            print(f"    └─ Information Density: {np.mean(overall_metrics['information_density']):.4f} ± {np.std(overall_metrics['information_density']):.4f}")
            
            print(f"\n  ⚡ EFFICIENCY METRICS:")
            print(f"    ├─ Compression: {np.mean(overall_metrics['compression_ratio']):.2f}x ± {np.std(overall_metrics['compression_ratio']):.2f}x")
            print(f"    ├─ Total Bandwidth Saved: {sum(overall_metrics['bits_saved'])/1e6:.2f} Mb")
            print(f"    ├─ Avg Latency: {np.mean(overall_metrics['inference_time']):.3f}s ± {np.std(overall_metrics['inference_time']):.3f}s")
            print(f"    └─ Total Samples: {len(overall_metrics['answer_completeness'])}")
            
            # Composite score
            quality_score = np.mean([
                np.mean(overall_metrics['contextual_relevance']),
                np.mean(overall_metrics['answer_completeness']),
                np.mean(overall_metrics['response_coherence']),
                np.mean(overall_metrics['information_density']),
            ])
            
            efficiency_score = min(1.0, np.mean(overall_metrics['compression_ratio']) / 3.0)
            composite_score = 0.6 * quality_score + 0.4 * efficiency_score
            
            print(f"\n  🏆 COMPOSITE SCORE: {composite_score:.4f}")
            print(f"    ├─ Quality (60%): {quality_score:.4f}")
            print(f"    └─ Efficiency (40%): {efficiency_score:.4f}")
            
            # Add interpretation
            if composite_score > 0.7:
                rating = "EXCELLENT ⭐⭐⭐⭐⭐"
            elif composite_score > 0.6:
                rating = "GOOD ⭐⭐⭐⭐"
            elif composite_score > 0.5:
                rating = "ACCEPTABLE ⭐⭐⭐"
            else:
                rating = "NEEDS IMPROVEMENT ⭐⭐"
            
            print(f"\n  Overall Rating: {rating}")