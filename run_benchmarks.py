#!/usr/bin/env python3
"""
CVL Benchmark Runner
====================
Simple script to run comprehensive benchmarks for:
1. Compressed Vector Language (CVL) compression efficiency
2. Semantic preservation testing
3. 10 Task-specific evaluations

Usage:
    python run_benchmarks.py              # Full benchmarks (~2-3 minutes)
    python run_benchmarks.py --quick      # Quick benchmarks (~1 minute)
    python run_benchmarks.py --help       # Show help
"""

import argparse
import json
import sys
import time
from cvl_benchmark_suite import CVLBenchmarkSuite
from unsupervised_cvl import UnsupervisedCVL
from real_data_generator import RealAgentDataGenerator


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║           CVL COMPREHENSIVE BENCHMARK SUITE                       ║
    ║                                                                   ║
    ║        Testing Compressed Vector Language Performance            ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive CVL benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py              # Run full benchmarks
  python run_benchmarks.py --quick      # Run quick benchmarks
  python run_benchmarks.py --samples 500  # Use 500 messages
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks with fewer samples (faster)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of messages to generate (default: 1000 for full, 300 for quick)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='cvl_benchmark_results.json',
        help='Output file for results (default: cvl_benchmark_results.json)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip CVL training if model exists (for testing)'
    )
    
    return parser.parse_args()


def generate_dataset(num_samples: int) -> list:
    """Generate agent communication dataset"""
    print("=" * 70)
    print("STEP 1: GENERATING AGENT COMMUNICATION DATASET")
    print("=" * 70)
    
    print(f"\nGenerating {num_samples} realistic agent messages...")
    generator = RealAgentDataGenerator()
    
    start_time = time.time()
    messages = generator.generate_dataset(num_samples)
    generation_time = time.time() - start_time
    
    # Get statistics
    stats = generator.get_message_statistics(messages)
    
    print(f"\n✓ Generated {len(messages)} messages in {generation_time:.1f}s")
    print(f"  • Avg message size: {stats['avg_json_size']:.0f} bytes")
    print(f"  • Total dataset: {stats['total_json_size']:,} bytes ({stats['total_json_size']/1024:.1f} KB)")
    print(f"  • Message types: {len(stats['type_distribution'])}")
    
    # Show distribution
    print(f"\n  Message Type Distribution:")
    for msg_type, count in sorted(stats['type_distribution'].items()):
        percentage = (count / stats['total_messages']) * 100
        bar = '█' * int(percentage / 2)
        print(f"    {msg_type:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    return messages


def train_cvl_model(messages: list, skip_training: bool = False) -> UnsupervisedCVL:
    """Initialize and train CVL model"""
    print("\n" + "=" * 70)
    print("STEP 2: INITIALIZING & TRAINING CVL MODEL")
    print("=" * 70)
    
    print("\nInitializing CVL system...")
    cvl = UnsupervisedCVL(bandwidth_budget=1000.0)
    
    if not skip_training:
        print("Training CVL model (this may take a minute)...")
        start_time = time.time()
        
        training_stats = cvl.fit_unsupervised(messages)
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.1f}s")
        print(f"  • Embedding dimension: {training_stats['embedding_dim']}D")
        print(f"  • Compressed dimension: {training_stats['compressed_dim']}D")
        print(f"  • Compression: {training_stats['embedding_dim']}D → {training_stats['compressed_dim']}D")
        print(f"  • Variance explained: {training_stats['explained_variance']:.3f}")
        print(f"  • Message types learned: {training_stats['message_types']}")
        print(f"  • Priority levels: {training_stats['priorities']}")
    else:
        print("⚠ Skipping training (using default model)")
    
    return cvl


def run_benchmarks(cvl: UnsupervisedCVL, messages: list, quick_mode: bool = False) -> dict:
    """Run comprehensive benchmark suite"""
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING COMPREHENSIVE BENCHMARKS")
    print("=" * 70)
    
    if quick_mode:
        print("\n⚡ Quick Mode: Using reduced sample sizes for faster execution")
    else:
        print("\n🔬 Full Mode: Running complete benchmarks (this will take 3-5 minutes)")
    
    print("\nBenchmark Components:")
    print("  1. Compression Ratio & Performance")
    print("  2. Semantic Preservation")
    print("  3. Task-Specific Performance (10 tasks)")
    print("  4. Truth Token System")
    print("  5. Truth Token + CVL Integration")
    print("  6. Overall System Evaluation")
    
    print("\nStarting benchmarks...\n")
    
    # Initialize and run benchmark suite
    benchmark_suite = CVLBenchmarkSuite(cvl_model=cvl)
    results = benchmark_suite.run_all_benchmarks(messages, quick_mode=quick_mode)
    
    return results, benchmark_suite


def save_results(results: dict, benchmark_suite, output_file: str):
    """Save benchmark results to file"""
    print("\n" + "=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70)
    
    print(f"\nSaving results to {output_file}...")
    success = benchmark_suite.save_results(output_file)
    
    if success:
        print(f"✓ Results successfully saved")
        print(f"\nYou can view the results by opening: {output_file}")
    else:
        print(f"❌ Error saving results")


def print_quick_summary(results: dict):
    """Print a quick summary of key metrics"""
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    
    overall = results.get('overall_metrics', {})
    compression = results.get('compression_benchmarks', {})
    semantic = results.get('semantic_preservation', {})
    
    print(f"""
📊 Overall CVL Score:        {overall.get('overall_cvl_score', 0):.1f}/100
📈 Letter Grade:             {overall.get('letter_grade', 'N/A')}
⭐ Performance Rating:       {overall.get('performance_rating', 'N/A')}

🗜️  Compression Metrics:
   • Compression Ratio:      {compression.get('compression_ratio', 0):.1f}x
   • Space Savings:          {compression.get('space_savings_percent', 0):.1f}%
   • Avg Compression Time:   {compression.get('avg_compression_time_ms', 0):.2f}ms
   • Throughput:             {compression.get('throughput_messages_per_sec', 0):.1f} msg/sec

🔍 Semantic Preservation:
   • Cosine Similarity:      {semantic.get('avg_cosine_similarity', 0):.3f}
   • Type Preservation:      {semantic.get('type_preservation_accuracy', 0):.1%}
   • Overall Semantic:       {semantic.get('overall_semantic_score', 0):.3f}

⏱️  Total Time:              {results.get('benchmark_metadata', {}).get('total_execution_time_seconds', 0):.1f}s
    """)


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print_banner()
    
    # Determine sample size
    if args.samples:
        num_samples = args.samples
    elif args.quick:
        num_samples = 300
    else:
        num_samples = 1000
    
    print(f"\nConfiguration:")
    print(f"  • Mode: {'Quick' if args.quick else 'Full'}")
    print(f"  • Sample size: {num_samples} messages")
    print(f"  • Output file: {args.output}")
    print(f"  • Skip training: {'Yes' if args.skip_training else 'No'}")
    
    try:
        # Step 1: Generate dataset
        messages = generate_dataset(num_samples)
        
        # Step 2: Train CVL model
        cvl = train_cvl_model(messages, skip_training=args.skip_training)
        
        # Step 3: Run benchmarks
        results, benchmark_suite = run_benchmarks(cvl, messages, quick_mode=args.quick)
        
        # Step 4: Save results
        save_results(results, benchmark_suite, args.output)
        
        # Print quick summary
        print_quick_summary(results)
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Return success
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

