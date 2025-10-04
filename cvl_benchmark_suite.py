"""
Comprehensive Benchmark Suite for CVL System
Tests Compressed Vector Language compression efficiency and task performance
"""

import numpy as np
import json
import time
import random
from typing import Dict, List, Any, Tuple
from unsupervised_cvl import UnsupervisedCVL, CompressedAgenticMessage
from task_datasets import TaskDatasetGenerator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CVLBenchmarkSuite:
    """
    Comprehensive benchmark suite for CVL system
    Tests compression efficiency, semantic preservation, and task performance
    """
    
    def __init__(self, cvl_model: UnsupervisedCVL = None):
        print("=" * 70)
        print("INITIALIZING CVL BENCHMARK SUITE")
        print("=" * 70)
        
        self.cvl = cvl_model if cvl_model else UnsupervisedCVL()
        self.task_generator = TaskDatasetGenerator()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.results = {
            'compression_benchmarks': {},
            'task_benchmarks': {},
            'overall_metrics': {},
            'timestamp': time.time()
        }
        
        print("✓ CVL model loaded")
        print("✓ Task Generator ready")
        print("✓ Sentence Transformer loaded")
    
    # ==================== PART 1: COMPRESSION BENCHMARKS ====================
    
    def benchmark_compression_ratio(self, messages: List[Dict]) -> Dict[str, float]:
        """
        Benchmark compression ratio and performance metrics
        Tests: ratio, speed, throughput, space savings
        """
        print("\n" + "=" * 70)
        print("[BENCHMARK 1/4] COMPRESSION RATIO & PERFORMANCE")
        print("=" * 70)
        
        original_sizes = []
        compressed_sizes = []
        compression_times = []
        decompression_times = []
        successful_compressions = 0
        
        test_messages = messages[:100]  # Use 100 messages for speed
        
        print(f"Testing on {len(test_messages)} messages...")
        
        for i, msg in enumerate(test_messages):
            try:
                # Measure compression
                start = time.time()
                compressed = self.cvl.compress_message(msg)
                comp_time = time.time() - start
                compression_times.append(comp_time)
                
                # Measure decompression
                start = time.time()
                decompressed = self.cvl.decompress_message(compressed)
                decomp_time = time.time() - start
                decompression_times.append(decomp_time)
                
                # Calculate sizes
                original_size = len(json.dumps(msg).encode('utf-8'))
                compressed_size = len(compressed.to_bytes())
                
                original_sizes.append(original_size)
                compressed_sizes.append(compressed_size)
                successful_compressions += 1
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(test_messages)}...")
                
            except Exception as e:
                print(f"  Warning: Error compressing message {i}: {e}")
                continue
        
        # Calculate metrics
        avg_original = np.mean(original_sizes) if original_sizes else 0
        avg_compressed = np.mean(compressed_sizes) if compressed_sizes else 1
        
        results = {
            'avg_original_size_bytes': avg_original,
            'avg_compressed_size_bytes': avg_compressed,
            'compression_ratio': avg_original / avg_compressed if avg_compressed > 0 else 0,
            'space_savings_percent': (1 - avg_compressed / avg_original) * 100 if avg_original > 0 else 0,
            'avg_compression_time_ms': np.mean(compression_times) * 1000 if compression_times else 0,
            'avg_decompression_time_ms': np.mean(decompression_times) * 1000 if decompression_times else 0,
            'total_compression_time_ms': sum(compression_times) * 1000,
            'throughput_messages_per_sec': 1.0 / np.mean(compression_times) if compression_times and np.mean(compression_times) > 0 else 0,
            'success_rate': successful_compressions / len(test_messages) if test_messages else 0,
            'messages_tested': len(test_messages)
        }
        
        # Print results
        print(f"\n{'Metric':<40} {'Value':>15}")
        print("-" * 70)
        print(f"{'Average Original Size':<40} {results['avg_original_size_bytes']:>12.1f} bytes")
        print(f"{'Average Compressed Size':<40} {results['avg_compressed_size_bytes']:>12.1f} bytes")
        print(f"{'Compression Ratio':<40} {results['compression_ratio']:>14.2f}x")
        print(f"{'Space Savings':<40} {results['space_savings_percent']:>13.1f}%")
        print(f"{'Avg Compression Time':<40} {results['avg_compression_time_ms']:>12.2f}ms")
        print(f"{'Avg Decompression Time':<40} {results['avg_decompression_time_ms']:>12.2f}ms")
        print(f"{'Throughput':<40} {results['throughput_messages_per_sec']:>10.1f} msg/sec")
        print(f"{'Success Rate':<40} {results['success_rate']:>13.1%}")
        
        return results
    
    def benchmark_semantic_preservation(self, messages: List[Dict], num_samples: int = 50) -> Dict[str, float]:
        """
        Benchmark how well semantic meaning is preserved after compression
        Tests: cosine similarity, metadata preservation, embedding quality
        """
        print("\n" + "=" * 70)
        print("[BENCHMARK 2/4] SEMANTIC PRESERVATION")
        print("=" * 70)
        
        similarities = []
        type_accuracies = []
        priority_accuracies = []
        embedding_distances = []
        
        sample_msgs = messages[:num_samples]
        print(f"Testing semantic preservation on {len(sample_msgs)} messages...")
        
        for i, msg in enumerate(sample_msgs):
            try:
                # Get original embedding
                orig_content = msg['content']
                orig_embedding = self.sentence_model.encode([orig_content])[0]
                
                # Compress and decompress
                compressed = self.cvl.compress_message(msg)
                decompressed = self.cvl.decompress_message(compressed)
                
                # Embedding similarity (if decoded embedding available)
                if 'decoded_embedding' in decompressed:
                    dec_emb = np.array(decompressed['decoded_embedding'])
                    
                    # Ensure same dimensionality
                    if len(dec_emb) == len(orig_embedding):
                        similarity = cosine_similarity(
                            orig_embedding.reshape(1, -1),
                            dec_emb.reshape(1, -1)
                        )[0][0]
                        similarities.append(similarity)
                        
                        # Euclidean distance
                        distance = np.linalg.norm(orig_embedding - dec_emb)
                        embedding_distances.append(distance)
                
                # Metadata preservation
                type_match = 1.0 if decompressed.get('message_type') == msg.get('message_type') else 0.0
                priority_match = 1.0 if decompressed.get('priority') == msg.get('priority') else 0.0
                
                type_accuracies.append(type_match)
                priority_accuracies.append(priority_match)
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(sample_msgs)}...")
                
            except Exception as e:
                print(f"  Warning: Error processing message {i}: {e}")
                continue
        
        # Calculate results
        results = {
            'avg_cosine_similarity': np.mean(similarities) if similarities else 0.0,
            'min_cosine_similarity': np.min(similarities) if similarities else 0.0,
            'max_cosine_similarity': np.max(similarities) if similarities else 0.0,
            'std_cosine_similarity': np.std(similarities) if similarities else 0.0,
            'avg_embedding_distance': np.mean(embedding_distances) if embedding_distances else 0.0,
            'type_preservation_accuracy': np.mean(type_accuracies) if type_accuracies else 0.0,
            'priority_preservation_accuracy': np.mean(priority_accuracies) if priority_accuracies else 0.0,
            'overall_semantic_score': 0.0,
            'samples_tested': len(sample_msgs),
            'successful_comparisons': len(similarities)
        }
        
        # Calculate overall semantic score
        if similarities and type_accuracies:
            results['overall_semantic_score'] = np.mean([
                np.mean(similarities),
                np.mean(type_accuracies),
                np.mean(priority_accuracies)
            ])
        
        # Print results
        print(f"\n{'Metric':<40} {'Value':>15}")
        print("-" * 70)
        print(f"{'Avg Cosine Similarity':<40} {results['avg_cosine_similarity']:>15.3f}")
        print(f"{'Similarity Range':<40} {results['min_cosine_similarity']:.3f} - {results['max_cosine_similarity']:.3f}")
        print(f"{'Type Preservation Accuracy':<40} {results['type_preservation_accuracy']:>14.1%}")
        print(f"{'Priority Preservation Accuracy':<40} {results['priority_preservation_accuracy']:>14.1%}")
        print(f"{'Overall Semantic Score':<40} {results['overall_semantic_score']:>15.3f}")
        print(f"{'Successful Comparisons':<40} {results['successful_comparisons']:>10}/{results['samples_tested']}")
        
        return results
    
    # ==================== PART 2: TASK-SPECIFIC BENCHMARKS ====================
    
    def benchmark_task_performance(self, task_type: str, task_data: List[Dict], max_samples: int = 30) -> Dict[str, Any]:
        """
        Benchmark CVL performance on a specific task type
        Tests: compression on task-specific data, preservation of task-relevant info
        """
        print(f"\n  Testing: {task_type.replace('_', ' ').title()}", end='')
        
        results = {
            'task_type': task_type,
            'total_samples': len(task_data),
            'samples_tested': 0,
            'compressed_accurately': 0,
            'avg_compression_ratio': 0.0,
            'avg_compression_time_ms': 0.0,
            'task_specific_metrics': {}
        }
        
        compression_ratios = []
        compression_times = []
        samples_to_test = task_data[:max_samples]
        
        for item in samples_to_test:
            try:
                # Convert task item to CVL message format
                content = self._extract_content_from_task(task_type, item)
                
                msg = {
                    'content': content,
                    'message_type': task_type,
                    'priority': 'normal',
                    'timestamp': time.time()
                }
                
                # Compress
                start = time.time()
                compressed = self.cvl.compress_message(msg)
                comp_time = (time.time() - start) * 1000  # ms
                compression_times.append(comp_time)
                
                # Calculate compression ratio
                original_size = len(json.dumps(msg).encode('utf-8'))
                compressed_size = len(compressed.to_bytes())
                ratio = original_size / compressed_size if compressed_size > 0 else 0
                compression_ratios.append(ratio)
                
                # Decompress and check preservation
                decompressed = self.cvl.decompress_message(compressed)
                
                if decompressed.get('message_type') == task_type:
                    results['compressed_accurately'] += 1
                
                results['samples_tested'] += 1
                
            except Exception as e:
                continue
        
        # Calculate metrics
        results['avg_compression_ratio'] = np.mean(compression_ratios) if compression_ratios else 0.0
        results['avg_compression_time_ms'] = np.mean(compression_times) if compression_times else 0.0
        results['accuracy_rate'] = results['compressed_accurately'] / results['samples_tested'] if results['samples_tested'] > 0 else 0.0
        results['min_compression_ratio'] = np.min(compression_ratios) if compression_ratios else 0.0
        results['max_compression_ratio'] = np.max(compression_ratios) if compression_ratios else 0.0
        
        print(f" → Ratio: {results['avg_compression_ratio']:.1f}x, Accuracy: {results['accuracy_rate']:.1%}")
        
        return results
    
    def _extract_content_from_task(self, task_type: str, item: Dict) -> str:
        """Extract appropriate content string from task item"""
        if task_type == 'arithmetic':
            return item.get('question', '')
        elif task_type == 'summarization':
            return item.get('passage', '')
        elif task_type == 'paraphrasing':
            return item.get('original', '')
        elif task_type == 'sentence_completion':
            return item.get('sentence', '')
        elif task_type == 'classification':
            return item.get('text', '')
        elif task_type == 'translation':
            return item.get('source_text', '')
        elif task_type == 'qa_factual':
            return item.get('question', '')
        elif task_type == 'commonsense':
            return item.get('question', '')
        elif task_type == 'analogies':
            return item.get('question', '')
        elif task_type == 'entity_extraction':
            return item.get('text', '')
        else:
            return str(item)
    
    def benchmark_all_tasks(self, samples_per_task: int = 30) -> Dict[str, Dict]:
        """
        Run benchmarks on all 10 task types
        """
        print("\n" + "=" * 70)
        print("[BENCHMARK 3/4] TASK-SPECIFIC PERFORMANCE (10 Tasks)")
        print("=" * 70)
        
        print("Generating task datasets...")
        task_datasets = self.task_generator.generate_all_tasks(samples_per_task)
        print(f"✓ Generated {len(task_datasets)} task types\n")
        
        all_task_results = {}
        
        for task_name, task_data in task_datasets.items():
            task_results = self.benchmark_task_performance(task_name, task_data, max_samples=samples_per_task)
            all_task_results[task_name] = task_results
        
        # Calculate aggregate metrics
        avg_ratio = np.mean([r['avg_compression_ratio'] for r in all_task_results.values()])
        avg_accuracy = np.mean([r['accuracy_rate'] for r in all_task_results.values()])
        avg_time = np.mean([r['avg_compression_time_ms'] for r in all_task_results.values()])
        
        print(f"\n{'Overall Task Performance':<40}")
        print("-" * 70)
        print(f"{'Avg Compression Ratio (all tasks)':<40} {avg_ratio:>14.2f}x")
        print(f"{'Avg Accuracy Rate (all tasks)':<40} {avg_accuracy:>14.1%}")
        print(f"{'Avg Compression Time (all tasks)':<40} {avg_time:>12.2f}ms")
        
        return all_task_results
    
    # ==================== PART 3: OVERALL EVALUATION ====================
    
    def calculate_overall_score(self) -> Dict[str, Any]:
        """
        Calculate overall CVL system score (0-100) with letter grade
        """
        print("\n" + "=" * 70)
        print("[BENCHMARK 4/4] OVERALL SYSTEM EVALUATION")
        print("=" * 70)
        
        compression = self.results['compression_benchmarks']
        semantic = self.results['semantic_preservation']
        tasks = self.results['task_benchmarks']
        
        # Component scores (0-100)
        # Compression score: ratio >= 20x = 100, linear scale
        compression_score = min(100, (compression['compression_ratio'] / 20.0) * 100)
        
        # Semantic score: based on similarity and preservation
        semantic_score = semantic['overall_semantic_score'] * 100
        
        # Task performance score: average accuracy across tasks
        task_accuracies = [t['accuracy_rate'] for t in tasks.values()]
        task_score = np.mean(task_accuracies) * 100 if task_accuracies else 0.0
        
        # Weighted overall score (without truth tokens for now)
        overall_score = (
            compression_score * 0.40 +  # 40% weight
            semantic_score * 0.30 +      # 30% weight
            task_score * 0.30            # 30% weight
        )
        
        # Letter grade
        grade = self._calculate_grade(overall_score)
        
        # Performance rating
        if overall_score >= 85:
            rating = "Excellent"
        elif overall_score >= 75:
            rating = "Good"
        elif overall_score >= 65:
            rating = "Satisfactory"
        elif overall_score >= 50:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        overall_metrics = {
            'overall_cvl_score': overall_score,
            'compression_score': compression_score,
            'semantic_score': semantic_score,
            'task_performance_score': task_score,
            'letter_grade': grade,
            'performance_rating': rating,
            'timestamp': time.time()
        }
        
        # Print results
        print(f"\n{'Component':<40} {'Score':>10}")
        print("-" * 70)
        print(f"{'Compression Performance':<40} {compression_score:>9.1f}/100")
        print(f"{'Semantic Preservation':<40} {semantic_score:>9.1f}/100")
        print(f"{'Task Performance':<40} {task_score:>9.1f}/100")
        print("=" * 70)
        print(f"{'OVERALL CVL SCORE':<40} {overall_score:>9.1f}/100")
        print(f"{'LETTER GRADE':<40} {grade:>10}")
        print(f"{'PERFORMANCE RATING':<40} {rating:>10}")
        
        return overall_metrics
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 93: return 'A'
        elif score >= 90: return 'A-'
        elif score >= 87: return 'B+'
        elif score >= 83: return 'B'
        elif score >= 80: return 'B-'
        elif score >= 77: return 'C+'
        elif score >= 73: return 'C'
        elif score >= 70: return 'C-'
        elif score >= 67: return 'D+'
        elif score >= 63: return 'D'
        elif score >= 60: return 'D-'
        else: return 'F'
    
    # ==================== MAIN EXECUTION ====================
    
    def run_all_benchmarks(self, messages: List[Dict] = None, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run complete benchmark suite
        
        Args:
            messages: List of agent messages (generated if None)
            quick_mode: If True, use fewer samples for faster execution
        
        Returns:
            Complete benchmark results dictionary
        """
        print("\n" + "=" * 70)
        print("STARTING COMPREHENSIVE CVL BENCHMARK SUITE")
        print("=" * 70)
        
        start_time = time.time()
        
        # Generate or prepare messages
        if messages is None:
            print("\nGenerating test messages...")
            from real_data_generator import RealAgentDataGenerator
            generator = RealAgentDataGenerator()
            num_messages = 300 if quick_mode else 1000
            messages = generator.generate_dataset(num_messages)
            print(f"✓ Generated {len(messages)} test messages")
        
        # Train CVL if not already trained
        if not hasattr(self.cvl, 'training_stats') or self.cvl.training_stats is None:
            print("\nTraining CVL model...")
            training_start = time.time()
            self.cvl.fit_unsupervised(messages)
            training_time = time.time() - training_start
            print(f"✓ Training completed in {training_time:.1f} seconds")
        
        # Set sample sizes based on mode
        compression_samples = 50 if quick_mode else 100
        semantic_samples = 25 if quick_mode else 50
        task_samples = 15 if quick_mode else 30
        
        # Run benchmark components
        try:
            # 1. Compression benchmarks
            compression_results = self.benchmark_compression_ratio(messages[:compression_samples])
            self.results['compression_benchmarks'] = compression_results
            
            # 2. Semantic preservation
            semantic_results = self.benchmark_semantic_preservation(messages[:semantic_samples])
            self.results['semantic_preservation'] = semantic_results
            
            # 3. Task-specific benchmarks
            task_results = self.benchmark_all_tasks(samples_per_task=task_samples)
            self.results['task_benchmarks'] = task_results
            
            # 4. Overall evaluation
            overall_metrics = self.calculate_overall_score()
            self.results['overall_metrics'] = overall_metrics
            
        except Exception as e:
            print(f"\n❌ Error during benchmarking: {e}")
            import traceback
            traceback.print_exc()
            return self.results
        
        # Final summary
        total_time = time.time() - start_time
        self.results['benchmark_metadata'] = {
            'total_execution_time_seconds': total_time,
            'messages_used': len(messages),
            'quick_mode': quick_mode,
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._print_final_summary(total_time)
        
        return self.results
    
    def _print_final_summary(self, execution_time: float):
        """Print final benchmark summary"""
        print("\n" + "=" * 70)
        print("BENCHMARK SUITE COMPLETE")
        print("=" * 70)
        
        overall = self.results['overall_metrics']
        compression = self.results['compression_benchmarks']
        semantic = self.results['semantic_preservation']
        
        print(f"\n{'='*70}")
        print(f"{'🏆 FINAL RESULTS':^70}")
        print(f"{'='*70}")
        print(f"\n{'Overall CVL Score:':<40} {overall['overall_cvl_score']:>9.1f}/100")
        print(f"{'Letter Grade:':<40} {overall['letter_grade']:>10}")
        print(f"{'Performance Rating:':<40} {overall['performance_rating']:>10}")
        print(f"\n{'Key Metrics:'}")
        print(f"  • Compression Ratio: {compression['compression_ratio']:.1f}x")
        print(f"  • Space Savings: {compression['space_savings_percent']:.1f}%")
        print(f"  • Compression Speed: {compression['avg_compression_time_ms']:.2f}ms")
        print(f"  • Semantic Similarity: {semantic['avg_cosine_similarity']:.3f}")
        print(f"  • Type Preservation: {semantic['type_preservation_accuracy']:.1%}")
        print(f"\n{'Total Execution Time:':<40} {execution_time:>12.1f}s")
        
        # Recommendations
        print(f"\n{'RECOMMENDATIONS:'}")
        score = overall['overall_cvl_score']
        if score >= 85:
            print("  ✓ Excellent performance! CVL system is production-ready.")
            print("  ✓ Compression and semantic preservation are working well.")
        elif score >= 75:
            print("  → Good performance overall.")
            if overall['compression_score'] < 75:
                print("  → Consider optimizing compression ratio.")
            if overall['semantic_score'] < 75:
                print("  → Consider improving semantic preservation.")
        elif score >= 65:
            print("  ⚠ Satisfactory but room for improvement.")
            print("  → Focus on semantic preservation and task performance.")
        else:
            print("  ⚠ Needs significant improvement.")
            print("  → Review compression algorithms and semantic preservation.")
        
        print("=" * 70)
    
    def save_results(self, filepath: str = "cvl_benchmark_results.json"):
        """Save complete benchmark results to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"\n✓ Results saved to {filepath} ({file_size:.1f} KB)")
            return True
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")
            return False


import os

if __name__ == "__main__":
    print("CVL Benchmark Suite - Direct Execution")
    print("Use run_benchmarks.py for full demonstration")

