#!/usr/bin/env python3
"""
Verification script to check if all benchmark components are properly installed
Run this before running benchmarks to ensure everything is set up correctly
"""

import sys
import importlib

def check_file_exists(filepath, description):
    """Check if a file exists"""
    import os
    if os.path.exists(filepath):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - NOT FOUND")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - {str(e)}")
        return False

def main():
    print("=" * 70)
    print("CVL BENCHMARK SUITE - INSTALLATION VERIFICATION")
    print("=" * 70)
    
    all_checks_passed = True
    
    # Check files
    print("\n📁 Checking Files...")
    files_to_check = [
        ("truth_token_system.py", "Truth Token System"),
        ("task_datasets.py", "Task Dataset Generator"),
        ("cvl_benchmark_suite.py", "Benchmark Suite"),
        ("run_benchmarks.py", "Runner Script"),
        ("unsupervised_cvl.py", "CVL System"),
        ("real_data_generator.py", "Data Generator"),
        ("BENCHMARKING_GUIDE.md", "Benchmarking Guide"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
        ("QUICK_START.md", "Quick Start Guide"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check Python dependencies
    print("\n📦 Checking Python Dependencies...")
    dependencies = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    for module, description in dependencies:
        if not check_import(module, description):
            all_checks_passed = False
    
    # Check custom modules
    print("\n🔧 Checking Custom Modules...")
    custom_modules = [
        ("truth_token_system", "Truth Token System Module"),
        ("task_datasets", "Task Dataset Module"),
        ("cvl_benchmark_suite", "Benchmark Suite Module"),
        ("unsupervised_cvl", "CVL System Module"),
        ("real_data_generator", "Data Generator Module"),
    ]
    
    for module, description in custom_modules:
        if not check_import(module, description):
            all_checks_passed = False
    
    # Try importing key classes
    print("\n🎯 Checking Key Classes...")
    try:
        from truth_token_system import TruthTokenSystem, TruthToken
        print("✅ TruthTokenSystem and TruthToken")
    except Exception as e:
        print(f"❌ TruthTokenSystem and TruthToken - {str(e)}")
        all_checks_passed = False
    
    try:
        from task_datasets import TaskDatasetGenerator
        print("✅ TaskDatasetGenerator")
    except Exception as e:
        print(f"❌ TaskDatasetGenerator - {str(e)}")
        all_checks_passed = False
    
    try:
        from cvl_benchmark_suite import CVLBenchmarkSuite
        print("✅ CVLBenchmarkSuite")
    except Exception as e:
        print(f"❌ CVLBenchmarkSuite - {str(e)}")
        all_checks_passed = False
    
    try:
        from unsupervised_cvl import UnsupervisedCVL
        print("✅ UnsupervisedCVL")
    except Exception as e:
        print(f"❌ UnsupervisedCVL - {str(e)}")
        all_checks_passed = False
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\n🚀 You're ready to run benchmarks!")
        print("\nRun: python run_benchmarks.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n⚠️  Please fix the issues above before running benchmarks.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure all files are in the same directory")
        print("  3. Check Python version (3.8+ required)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

