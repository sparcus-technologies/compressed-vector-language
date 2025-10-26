"""Comprehensive benchmarking suite using real open-source datasets with improved metrics"""

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
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic metrics disabled.")
    print("Install with: pip install sentence-transformers")


class ImprovedMetrics:
    """Better metrics for Q-KVComm evaluation"""
    
    def __init__(self):
        """Initialize semantic similarity model if available"""
        self.semantic_model = None
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded semantic similarity model")
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}")
    
    def answer_contains_truth(self, prediction: str, ground_truth: str) -> float:
        """Check if prediction contains the ground truth answer (more lenient than F1)"""
        pred_lower = prediction.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # Exact substring match
        if truth_lower in pred_lower:
            return 1.0
        
        # Check if key words are present
        truth_tokens = set(truth_lower.split())
        pred_tokens = set(pred_lower.split())
        
        if not truth_tokens:
            return 0.0
        
        # Percentage of ground truth tokens present in prediction
        overlap = len(truth_tokens & pred_tokens) / len(truth_tokens)
        return overlap
    
    def semantic_similarity(self, prediction: str, ground_truth: str) -> float:
        """Compute semantic similarity using embeddings"""
        if self.semantic_model is None:
            return 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get embeddings
            pred_emb = self.semantic_model.encode([prediction])
            truth_emb = self.semantic_model.encode([ground_truth])
            
            # Cosine similarity
            similarity = cosine_similarity(pred_emb, truth_emb)[0][0]
            return float(similarity)
        except Exception as e:
            return 0.0
    
    def information_retention(self, prediction: str, context: str) -> float:
        """Measure if key information from context is retained in prediction"""
        if self.semantic_model is None or not context:
            return 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            pred_emb = self.semantic_model.encode([prediction])
            ctx_emb = self.semantic_model.encode([context[:500]])  # First 500 chars
            
            similarity = cosine_similarity(pred_emb, ctx_emb)[0][0]
            return float(similarity)
        except Exception as e:
            return 0.0
    
    def compute_all_metrics(
        self, 
        prediction: str, 
        ground_truth: str,
        context: str = ""
    ) -> dict:
        """Compute all improved metrics"""
        
        return {
            'answer_correctness': self.answer_contains_truth(prediction, ground_truth),
            'semantic_similarity': self.semantic_similarity(prediction, ground_truth),
            'information_retention': self.information_retention(prediction, context) if context else 0.0,
        }


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
        self.improved_metrics = ImprovedMetrics()
        
    def load_squad(self, max_samples: int = 100) -> List[Dict]:
        """Load SQuAD 2.0 dataset - FIXED to handle unanswerable questions"""
        print("Loading SQuAD 2.0 dataset...")
        dataset = load_dataset("squad_v2", split="validation")
        
        samples = []
        total_checked = 0
        
        for item in dataset:
            if len(samples) >= max_samples:
                break
            
            total_checked += 1
            
            # FIXED: Handle both answerable and unanswerable questions
            if not item['answers']['text']:
                # Unanswerable question - use placeholder
                samples.append({
                    'id': item['id'],
                    'context': item['context'],
                    'question': item['question'],
                    'answers': ["unanswerable"],  # Special marker
                    'dataset': 'squad',
                    'is_answerable': False
                })
            else:
                # Answerable question
                samples.append({
                    'id': item['id'],
                    'context': item['context'],
                    'question': item['question'],
                    'answers': item['answers']['text'],
                    'dataset': 'squad',
                    'is_answerable': True
                })
        
        answerable_count = sum(1 for s in samples if s.get('is_answerable', True))
        print(f"Loaded {len(samples)} SQuAD samples ({answerable_count} answerable, {len(samples)-answerable_count} unanswerable)")
        print(f"  Checked {total_checked} items to get {len(samples)} samples")
        
        return samples
    
    def load_hotpot_qa(self, max_samples: int = 100) -> List[Dict]:
        """Load HotpotQA dataset (multi-hop reasoning)"""
        print("Loading HotpotQA dataset...")
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
        except Exception as e:
            print(f"Warning: Could not load HotpotQA: {e}")
            return []
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:  # Check more items
                break
            if len(samples) >= max_samples:
                break
            
            # Concatenate supporting facts as context
            context_parts = []
            try:
                for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                    context_parts.extend(sentences)
                context = ' '.join(context_parts[:1000])  # Limit context length
                
                if context:  # Only add if we have context
                    samples.append({
                        'id': item['id'],
                        'context': context,
                        'question': item['question'],
                        'answers': [item['answer']],
                        'dataset': 'hotpot_qa',
                        'is_answerable': True
                    })
            except Exception as e:
                continue
        
        print(f"Loaded {len(samples)} HotpotQA samples")
        return samples
    
    def load_triviaqa(self, max_samples: int = 100) -> List[Dict]:
        """Load TriviaQA dataset"""
        print("Loading TriviaQA dataset...")
        try:
            dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        except Exception as e:
            print(f"Warning: Could not load TriviaQA: {e}")
            return []
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:
                break
            if len(samples) >= max_samples:
                break
            
            # Use search results as context
            try:
                context = item['search_results']['search_context'][0] if item['search_results']['search_context'] else ""
                
                if context:
                    samples.append({
                        'id': f"triviaqa_{i}",
                        'context': context[:1000],
                        'question': item['question'],
                        'answers': item['answer']['aliases'],
                        'dataset': 'triviaqa',
                        'is_answerable': True
                    })
            except Exception as e:
                continue
        
        print(f"Loaded {len(samples)} TriviaQA samples")
        return samples
    
    def load_coqa(self, max_samples: int = 100) -> List[Dict]:
        """Load CoQA dataset (conversational QA)"""
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
            
            # Use first question from each conversation
            if item['questions']:
                samples.append({
                    'id': f"coqa_{item['id']}",
                    'context': item['story'],
                    'question': item['questions'][0],
                    'answers': [item['answers']['input_text'][0]],
                    'dataset': 'coqa',
                    'is_answerable': True
                })
        
        print(f"Loaded {len(samples)} CoQA samples")
        return samples
    
    def load_msmarco(self, max_samples: int = 100) -> List[Dict]:
        """Load MS MARCO dataset"""
        print("Loading MS MARCO dataset...")
        try:
            dataset = load_dataset("ms_marco", "v2.1", split="validation")
        except Exception as e:
            print(f"Warning: Could not load MS MARCO: {e}")
            return []
        
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:
                break
            if len(samples) >= max_samples:
                break
            
            # Use passages as context
            try:
                if item['passages']['passage_text']:
                    context = ' '.join(item['passages']['passage_text'][:3])  # First 3 passages
                    
                    if context and item['answers']:
                        samples.append({
                            'id': f"msmarco_{i}",
                            'context': context[:1000],
                            'question': item['query'],
                            'answers': item['answers'],
                            'dataset': 'msmarco',
                            'is_answerable': True
                        })
            except Exception as e:
                continue
        
        print(f"Loaded {len(samples)} MS MARCO samples")
        return samples
    
    def compute_metrics(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """Compute traditional F1/EM metrics (kept for comparison)"""
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            import re
            import string
            
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            
            def white_space_fix(text):
                return ' '.join(text.split())
            
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        def f1_score(prediction, ground_truth):
            pred_tokens = normalize_answer(prediction).split()
            truth_tokens = normalize_answer(ground_truth).split()
            
            if len(pred_tokens) == 0 or len(truth_tokens) == 0:
                return int(pred_tokens == truth_tokens)
            
            common_tokens = set(pred_tokens) & set(truth_tokens)
            
            if len(common_tokens) == 0:
                return 0
            
            prec = len(common_tokens) / len(pred_tokens)
            rec = len(common_tokens) / len(truth_tokens)
            
            return 2 * (prec * rec) / (prec + rec)
        
        def exact_match(prediction, ground_truth):
            return normalize_answer(prediction) == normalize_answer(ground_truth)
        
        # Compute metrics against all ground truths
        f1 = max(f1_score(prediction, gt) for gt in ground_truths)
        em = max(exact_match(prediction, gt) for gt in ground_truths)
        
        return {
            'f1': f1,
            'exact_match': em
        }
    
    def evaluate_sample(self, sample: Dict, max_new_tokens: int = 50) -> Dict:
        """Evaluate single sample with both traditional and improved metrics"""
        try:
            start_time = time.time()
            
            # Run Q-KVComm
            output, comm_metrics = self.qkvcomm.communicate(
                context=sample['context'],
                query=sample['question'],
                max_new_tokens=max_new_tokens
            )
            
            inference_time = time.time() - start_time
            
            # Traditional metrics (F1/EM - kept for comparison)
            traditional_metrics = self.compute_metrics(output, sample['answers'])
            
            # NEW: Improved metrics (more meaningful for Q-KVComm)
            improved = self.improved_metrics.compute_all_metrics(
                prediction=output,
                ground_truth=sample['answers'][0],
                context=sample['context']
            )
            
            result = {
                'id': sample['id'],
                'dataset': sample['dataset'],
                'question': sample['question'],
                'prediction': output,
                'ground_truth': sample['answers'][0] if sample['answers'] else '',
                'is_answerable': sample.get('is_answerable', True),
                
                # Traditional metrics (for comparison)
                'f1': traditional_metrics['f1'],
                'exact_match': traditional_metrics['exact_match'],
                
                # NEW: Improved metrics (primary evaluation)
                'answer_correctness': improved['answer_correctness'],
                'semantic_similarity': improved['semantic_similarity'],
                'information_retention': improved['information_retention'],
                
                # Communication metrics (MOST IMPORTANT for Q-KVComm!)
                'inference_time': inference_time,
                'compression_ratio': comm_metrics.get('avg_compression_ratio', 1.0),
                'layers_transmitted': comm_metrics.get('num_layers_transmitted', 0),
                'bits_saved': comm_metrics.get('total_bits_original', 0) - comm_metrics.get('total_bits_compressed', 0),
                'success': True
            }
            
        except Exception as e:
            result = {
                'id': sample['id'],
                'dataset': sample['dataset'],
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_benchmark(self, dataset_names: List[str], max_samples: int = 100, max_new_tokens: int = 50):
        """
        Run comprehensive benchmark
        
        Args:
            dataset_names: List of dataset names to benchmark
            max_samples: Max samples per dataset
            max_new_tokens: Max tokens to generate
        """
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not installed. Run: pip install datasets")
            return
        
        print("="*80)
        print("Q-KVCOMM COMPREHENSIVE BENCHMARK")
        print("="*80)
        print(f"\nDatasets: {dataset_names}")
        print(f"Max samples per dataset: {max_samples}")
        print(f"Max new tokens: {max_new_tokens}\n")
        
        # Load all datasets
        all_samples = []
        dataset_loaders = {
            'squad': self.load_squad,
            'hotpot_qa': self.load_hotpot_qa,
            'triviaqa': self.load_triviaqa,
            'coqa': self.load_coqa,
            'msmarco': self.load_msmarco,
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
        
        # Get calibration data (use first 50 or all if less)
        calibration_contexts = [s['context'] for s in all_samples[:min(50, len(all_samples))]]
        
        print(f"Using {len(calibration_contexts)} samples for calibration...")
        self.qkvcomm.calibrate(calibration_contexts)
        
        # Run evaluation
        print(f"\nEvaluating {len(all_samples)} samples...\n")
        
        for sample in tqdm(all_samples, desc="Evaluating"):
            result = self.evaluate_sample(sample, max_new_tokens)
            self.results[sample['dataset']].append(result)
        
        # Save and analyze results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        
        # Save as CSV
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
        """Print benchmark summary with improved metrics"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80 + "\n")
        
        overall_metrics = {
            # Traditional metrics
            'f1': [], 'exact_match': [],
            # Improved metrics
            'answer_correctness': [], 'semantic_similarity': [], 'information_retention': [],
            # Communication metrics
            'inference_time': [], 'compression_ratio': [], 'layers_transmitted': [], 'bits_saved': []
        }
        
        for dataset_name, results in self.results.items():
            successful = [r for r in results if r.get('success', False)]
            
            if not successful:
                continue
            
            # Separate answerable and unanswerable
            answerable = [r for r in successful if r.get('is_answerable', True)]
            unanswerable = [r for r in successful if not r.get('is_answerable', True)]
            
            print(f"\n{dataset_name.upper()}")
            print("-" * 40)
            print(f"  Total samples: {len(results)}")
            print(f"  Successful: {len(successful)}")
            if unanswerable:
                print(f"    - Answerable: {len(answerable)}")
                print(f"    - Unanswerable: {len(unanswerable)}")
            print(f"  Success rate: {len(successful)/len(results)*100:.1f}%")
            
            # Quality metrics - IMPROVED (only on answerable questions)
            eval_samples = answerable if answerable else successful
            
            avg_correctness = np.mean([r['answer_correctness'] for r in eval_samples])
            avg_semantic = np.mean([r.get('semantic_similarity', 0.0) for r in eval_samples])
            avg_retention = np.mean([r.get('information_retention', 0.0) for r in eval_samples])
            
            # Traditional metrics (for comparison)
            avg_f1 = np.mean([r['f1'] for r in eval_samples])
            avg_em = np.mean([r['exact_match'] for r in eval_samples])
            
            print(f"\n  Quality Metrics (IMPROVED) - on {len(eval_samples)} answerable:")
            print(f"    Answer Correctness: {avg_correctness:.4f} ⭐")
            if SEMANTIC_AVAILABLE:
                print(f"    Semantic Similarity: {avg_semantic:.4f} ⭐")
                print(f"    Information Retention: {avg_retention:.4f}")
            
            print(f"\n  Traditional Metrics (for comparison):")
            print(f"    F1 Score: {avg_f1:.4f}")
            print(f"    Exact Match: {avg_em:.4f}")
            
            # Communication metrics (MOST IMPORTANT!) - all samples
            avg_time = np.mean([r['inference_time'] for r in successful])
            avg_comp = np.mean([r['compression_ratio'] for r in successful])
            avg_layers = np.mean([r['layers_transmitted'] for r in successful])
            total_bits_saved = sum([r['bits_saved'] for r in successful])
            
            print(f"\n  Communication Metrics (PRIMARY) - all {len(successful)} samples:")
            print(f"    Avg Compression Ratio: {avg_comp:.2f}x ⭐⭐⭐")
            print(f"    Total Bits Saved: {total_bits_saved/1e6:.2f} Mb ⭐⭐⭐")
            print(f"    Avg Layers Transmitted: {avg_layers:.1f}")
            print(f"    Avg Inference Time: {avg_time:.3f}s")
            
            # Collect for overall statistics
            overall_metrics['f1'].extend([r['f1'] for r in eval_samples])
            overall_metrics['exact_match'].extend([r['exact_match'] for r in eval_samples])
            overall_metrics['answer_correctness'].extend([r['answer_correctness'] for r in eval_samples])
            overall_metrics['semantic_similarity'].extend([r.get('semantic_similarity', 0.0) for r in eval_samples])
            overall_metrics['information_retention'].extend([r.get('information_retention', 0.0) for r in eval_samples])
            overall_metrics['inference_time'].extend([r['inference_time'] for r in successful])
            overall_metrics['compression_ratio'].extend([r['compression_ratio'] for r in successful])
            overall_metrics['layers_transmitted'].extend([r['layers_transmitted'] for r in successful])
            overall_metrics['bits_saved'].extend([r['bits_saved'] for r in successful])
        
        # Overall statistics
        if overall_metrics['f1']:
            print("\n" + "="*80)
            print("OVERALL STATISTICS")
            print("="*80)
            
            print(f"\n  PRIMARY METRICS (Q-KVComm Contribution):")
            print(f"    Compression Ratio: {np.mean(overall_metrics['compression_ratio']):.2f}x (±{np.std(overall_metrics['compression_ratio']):.2f}x) ⭐⭐⭐")
            print(f"    Total Communication Saved: {sum(overall_metrics['bits_saved'])/1e6:.2f} Mb ⭐⭐⭐")
            print(f"    Answer Correctness: {np.mean(overall_metrics['answer_correctness']):.4f} (±{np.std(overall_metrics['answer_correctness']):.4f}) ⭐")
            
            if SEMANTIC_AVAILABLE:
                print(f"    Semantic Similarity: {np.mean(overall_metrics['semantic_similarity']):.4f} (±{np.std(overall_metrics['semantic_similarity']):.4f}) ⭐")
            
            print(f"\n  SECONDARY METRICS (for reference):")
            print(f"    Inference Time: {np.mean(overall_metrics['inference_time']):.3f}s (±{np.std(overall_metrics['inference_time']):.3f}s)")
            print(f"    Total Samples Evaluated: {len(overall_metrics['f1'])}")