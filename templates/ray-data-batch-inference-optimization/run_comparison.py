"""
Main Comparison Script - Ray Data Batch Inference Optimization

This script demonstrates the complete workflow:
1. Run baseline (poorly optimized) pipeline
2. Run optimized pipeline
3. Compare performance and generate detailed analysis
4. Provide optimization recommendations

Run this script to see the dramatic performance improvements possible with Ray Data optimization.
"""

import logging
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main comparison workflow"""
    logger.info("🚀 Ray Data Batch Inference Optimization - Performance Comparison")
    logger.info("="*70)
    
    try:
        # Import our modules
        logger.info("📦 Importing optimization modules...")
        from baseline_implementation import analyze_baseline_performance
        from optimized_implementation import analyze_optimized_performance
        from performance_analysis import run_comprehensive_analysis
        
        logger.info("✅ All modules imported successfully")
        
        # Check if Ray is available
        try:
            import ray
            logger.info(f"✅ Ray version: {ray.__version__}")
        except ImportError:
            logger.error("❌ Ray is not installed. Please install with: pip install 'ray[data]'")
            return 1
        
        # Check if PyTorch is available
        try:
            import torch
            logger.info(f"✅ PyTorch version: {torch.__version__}")
            logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA version: {torch.version.cuda}")
                logger.info(f"✅ GPU count: {torch.cuda.device_count()}")
        except ImportError:
            logger.error("❌ PyTorch is not installed. Please install with: pip install torch torchvision")
            return 1
        
        # Display configuration
        logger.info("\n🔧 CONFIGURATION:")
        logger.info("• Model: ResNet50 (ImageNet pretrained)")
        logger.info("• Dataset: 300 sample images (repeated for demonstration)")
        logger.info("• Baseline: Poorly optimized pipeline with common mistakes")
        logger.info("• Optimized: Best practices pipeline with 28+ optimizations")
        
        # Ask user for comparison method
        logger.info("\n📊 Choose comparison method:")
        logger.info("1. Quick comparison (run both pipelines separately)")
        logger.info("2. Comprehensive analysis (detailed performance metrics)")
        logger.info("3. Run baseline only (to see poor performance)")
        logger.info("4. Run optimized only (to see best practices)")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
        except KeyboardInterrupt:
            logger.info("\n👋 Comparison cancelled by user")
            return 0
        
        if choice == "1":
            logger.info("\n🚀 Running quick comparison...")
            run_quick_comparison(analyze_baseline_performance, analyze_optimized_performance)
            
        elif choice == "2":
            logger.info("\n🔍 Running comprehensive analysis...")
            run_comprehensive_analysis(analyze_baseline_performance, analyze_optimized_performance)
            
        elif choice == "3":
            logger.info("\n🐌 Running baseline pipeline only...")
            baseline_results = analyze_baseline_performance()
            logger.info("✅ Baseline pipeline completed")
            
        elif choice == "4":
            logger.info("\n✨ Running optimized pipeline only...")
            optimized_results = analyze_optimized_performance()
            logger.info("✅ Optimized pipeline completed")
            
        else:
            logger.error("❌ Invalid choice. Please run the script again and choose 1-4.")
            return 1
        
        logger.info("\n🎉 Performance comparison completed successfully!")
        logger.info("📁 Check the output files for detailed results")
        logger.info("📊 Review the console output for performance metrics")
        
        return 0
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please ensure all required packages are installed:")
        logger.error("pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error("Please check the error details and ensure your environment is properly configured")
        return 1

def run_quick_comparison(baseline_func, optimized_func):
    """Run a quick comparison of both pipelines"""
    logger.info("\n" + "="*60)
    logger.info("🚀 QUICK PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    # Run baseline
    logger.info("\n🐌 Running baseline pipeline...")
    baseline_results = baseline_func()
    
    # Run optimized
    logger.info("\n✨ Running optimized pipeline...")
    optimized_results = optimized_func()
    
    # Quick comparison
    logger.info("\n" + "="*60)
    logger.info("📊 QUICK COMPARISON RESULTS")
    logger.info("="*60)
    
    if baseline_results and optimized_results:
        speedup = baseline_results['total_time'] / optimized_results['total_time']
        throughput_improvement = optimized_results['throughput'] / baseline_results['throughput']
        
        logger.info(f"⏱️  Time Improvement: {speedup:.2f}x faster")
        logger.info(f"🚀 Throughput Improvement: {throughput_improvement:.2f}x higher")
        logger.info(f"💰 Efficiency Gain: {speedup:.2f}x more efficient")
        
        logger.info(f"\nBaseline Performance:")
        logger.info(f"  • Time: {baseline_results['total_time']:.2f} seconds")
        logger.info(f"  • Throughput: {baseline_results['throughput']:.2f} images/second")
        
        logger.info(f"\nOptimized Performance:")
        logger.info(f"  • Time: {optimized_results['total_time']:.2f} seconds")
        logger.info(f"  • Throughput: {optimized_results['throughput']:.2f} images/second")
        
        logger.info(f"\n🎯 Key Improvements:")
        logger.info(f"  • Time saved: {baseline_results['total_time'] - optimized_results['total_time']:.2f} seconds")
        logger.info(f"  • Throughput increase: +{((throughput_improvement-1)*100):.1f}%")
        
        if speedup > 3:
            logger.info("🎉 Excellent optimization! Pipeline is significantly faster")
        elif speedup > 2:
            logger.info("✅ Good optimization! Pipeline shows clear improvements")
        elif speedup > 1.5:
            logger.info("👍 Moderate optimization! Pipeline shows some improvements")
        else:
            logger.info("⚠️  Limited optimization. Check configuration and resources")
    
    else:
        logger.error("❌ Comparison failed - one or both pipelines did not complete successfully")

def print_optimization_summary():
    """Print a summary of the optimization techniques used"""
    logger.info("\n" + "="*60)
    logger.info("🎯 OPTIMIZATION TECHNIQUES SUMMARY")
    logger.info("="*60)
    
    optimizations = [
        "1. Proper batch size (128 vs 32) - optimal GPU utilization",
        "2. Scaled workers (8 vs 2) - better parallelism",
        "3. Using .map_batches() vs .map() - vectorized operations",
        "4. GPU allocation specification - proper resource usage",
        "5. Efficient preprocessing - direct numpy→tensor conversion",
        "6. Proper error handling - robust pipeline execution",
        "7. Output repartitioning - controlled output file sizes",
        "8. Streaming processing - reduced memory pressure",
        "9. Ray Data context optimization - better resource management",
        "10. Actor reuse and resource limits - improved efficiency",
        "11. Memory cleanup - GPU memory management",
        "12. Transform caching - avoid recreation",
        "13. Vectorized batch processing - process entire batch at once",
        "14. Proper resource allocation - CPU/GPU/memory limits",
        "15. Block size optimization - parallelism control",
        "16. Operator fusion - reduce data movement",
        "17. Progress monitoring - track pipeline health",
        "18. Efficient output handling - controlled file sizes",
        "19. Error recovery - graceful failure handling",
        "20. Resource monitoring - track utilization",
        "21. Batch format specification - numpy format efficiency",
        "22. Actor lifecycle management - proper cleanup",
        "23. Memory pressure monitoring - avoid OOM",
        "24. GPU memory optimization - batch size tuning",
        "25. Pipeline statistics - performance insights",
        "26. Resource contention avoidance - proper limits",
        "27. Streaming execution - memory efficiency",
        "28. Output optimization - file size control"
    ]
    
    for opt in optimizations:
        logger.info(f"  {opt}")
    
    logger.info("\n💡 These optimizations address the AI Complexity Wall by:")
    logger.info("  • Maximizing GPU utilization for AI workloads")
    logger.info("  • Efficient memory management for large datasets")
    logger.info("  • Proper resource allocation across heterogeneous compute")
    logger.info("  • Optimized data movement and processing patterns")

if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code == 0:
            print_optimization_summary()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n👋 Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
