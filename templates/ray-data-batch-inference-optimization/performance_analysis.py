"""
Performance Analysis Tools for Ray Data Pipelines

This module provides comprehensive tools for analyzing and optimizing Ray Data pipeline performance:
1. Pipeline diagnostics and bottleneck identification
2. Resource utilization analysis
3. Memory and GPU usage monitoring
4. Performance comparison and benchmarking
5. Optimization recommendations

Use these tools to identify performance issues and apply the right optimizations.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple
import ray
import numpy as np
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_time: float
    images_processed: int
    throughput: float
    memory_usage: Dict[str, float]
    gpu_usage: Optional[Dict[str, float]]
    resource_utilization: Dict[str, float]
    pipeline_stats: Optional[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]

class RayDataPerformanceAnalyzer:
    """
    Comprehensive performance analyzer for Ray Data pipelines
    """
    
    def __init__(self):
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.optimized_metrics: Optional[PerformanceMetrics] = None
        self.analysis_results: Dict[str, Any] = {}
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system resource usage"""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Ray cluster metrics
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()
                
                # Calculate resource utilization
                cpu_utilization = 1 - (available_resources.get('CPU', 0) / max(cluster_resources.get('CPU', 1), 1))
                memory_utilization = 1 - (available_resources.get('memory', 0) / max(cluster_resources.get('memory', 1), 1))
                gpu_utilization = 1 - (available_resources.get('GPU', 0) / max(cluster_resources.get('GPU', 1), 1))
            else:
                cpu_utilization = cpu_percent / 100
                memory_utilization = memory.percent / 100
                gpu_utilization = 0.0
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "ray_cpu_utilization": cpu_utilization,
                "ray_memory_utilization": memory_utilization,
                "ray_gpu_utilization": gpu_utilization,
                "ray_cluster_size": len(ray.nodes()) if ray.is_initialized() else 0
            }
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {}
    
    def collect_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Collect GPU utilization metrics if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpu_metrics = {}
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_metrics[f"gpu_{i}_memory_used_gb"] = info.used / (1024**3)
                gpu_metrics[f"gpu_{i}_memory_total_gb"] = info.total / (1024**3)
                gpu_metrics[f"gpu_{i}_utilization"] = utilization.gpu
                gpu_metrics[f"gpu_{i}_memory_utilization"] = (info.used / info.total) * 100
            
            return gpu_metrics
        except ImportError:
            logger.info("pynvml not available - GPU metrics will not be collected")
            return None
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            return None
    
    def analyze_pipeline_stats(self, dataset) -> Optional[Dict[str, Any]]:
        """Analyze Ray Data pipeline statistics if available"""
        try:
            # Get pipeline statistics
            stats = dataset.stats()
            
            # Parse stats for key metrics
            pipeline_metrics = {
                "total_blocks": getattr(stats, 'num_blocks', 0),
                "total_rows": getattr(stats, 'num_rows', 0),
                "operators": []
            }
            
            # Extract operator-level statistics
            if hasattr(stats, 'operators'):
                for op in stats.operators:
                    op_stats = {
                        "name": getattr(op, 'name', 'Unknown'),
                        "tasks_executed": getattr(op, 'tasks_executed', 0),
                        "blocks_produced": getattr(op, 'blocks_produced', 0),
                        "wall_time": getattr(op, 'wall_time', 0),
                        "cpu_time": getattr(op, 'cpu_time', 0),
                        "peak_memory": getattr(op, 'peak_memory', 0)
                    }
                    pipeline_metrics["operators"].append(op_stats)
            
            return pipeline_metrics
        except Exception as e:
            logger.warning(f"Failed to analyze pipeline stats: {e}")
            return None
    
    def run_performance_test(self, pipeline_func, test_name: str) -> PerformanceMetrics:
        """Run a performance test and collect comprehensive metrics"""
        logger.info(f"🔍 Running performance test: {test_name}")
        
        # Collect baseline system metrics
        start_metrics = self.collect_system_metrics()
        start_gpu_metrics = self.collect_gpu_metrics()
        
        # Start timing
        start_time = time.time()
        
        # Run the pipeline
        try:
            results = pipeline_func()
            
            # Collect end metrics
            end_time = time.time()
            end_metrics = self.collect_system_metrics()
            end_gpu_metrics = self.collect_gpu_metrics()
            
            # Calculate performance metrics
            total_time = end_time - start_time
            images_processed = results.get('images_processed', 0)
            throughput = images_processed / total_time if total_time > 0 else 0
            
            # Calculate resource utilization changes
            resource_utilization = {
                "cpu_utilization_change": end_metrics.get('ray_cpu_utilization', 0) - start_metrics.get('ray_cpu_utilization', 0),
                "memory_utilization_change": end_metrics.get('ray_memory_utilization', 0) - start_metrics.get('ray_memory_utilization', 0),
                "gpu_utilization_change": end_metrics.get('ray_gpu_utilization', 0) - start_metrics.get('ray_gpu_utilization', 0)
            }
            
            # Memory usage analysis
            memory_usage = {
                "start_memory_gb": start_metrics.get('memory_used_gb', 0),
                "end_memory_gb": end_metrics.get('memory_used_gb', 0),
                "memory_increase_gb": end_metrics.get('memory_used_gb', 0) - start_metrics.get('memory_used_gb', 0),
                "peak_memory_gb": max(start_metrics.get('memory_used_gb', 0), end_metrics.get('memory_used_gb', 0))
            }
            
            # GPU usage analysis
            gpu_usage = None
            if start_gpu_metrics and end_gpu_metrics:
                gpu_usage = {
                    "gpu_count": len([k for k in start_gpu_metrics.keys() if k.startswith('gpu_0')]),
                    "avg_gpu_utilization": np.mean([v for k, v in start_gpu_metrics.items() if 'utilization' in k and 'memory' not in k]),
                    "avg_memory_utilization": np.mean([v for k, v in start_gpu_metrics.items() if 'memory_utilization' in k])
                }
            
            # Pipeline statistics
            pipeline_stats = self.analyze_pipeline_stats(results.get('dataset', None)) if 'dataset' in results else None
            
            return PerformanceMetrics(
                total_time=total_time,
                images_processed=images_processed,
                throughput=throughput,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                resource_utilization=resource_utilization,
                pipeline_stats=pipeline_stats,
                errors=results.get('errors', []),
                warnings=results.get('warnings', [])
            )
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Performance test failed: {e}")
            
            return PerformanceMetrics(
                total_time=end_time - start_time,
                images_processed=0,
                throughput=0,
                memory_usage={},
                gpu_usage=None,
                resource_utilization={},
                pipeline_stats=None,
                errors=[str(e)],
                warnings=[]
            )
    
    def analyze_performance_issues(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance metrics and identify issues"""
        issues = {
            "critical": [],
            "warning": [],
            "info": [],
            "recommendations": []
        }
        
        # Analyze throughput
        if metrics.throughput < 10:  # Less than 10 images/second
            issues["critical"].append("Very low throughput - pipeline may be severely bottlenecked")
            issues["recommendations"].append("Check GPU utilization and batch size configuration")
        elif metrics.throughput < 50:
            issues["warning"].append("Low throughput - consider optimization")
            issues["recommendations"].append("Review batch size and worker configuration")
        
        # Analyze memory usage
        if metrics.memory_usage.get('memory_increase_gb', 0) > 8:  # More than 8GB increase
            issues["warning"].append("High memory usage increase - potential memory leaks")
            issues["recommendations"].append("Check for memory leaks in preprocessing or model inference")
        
        # Analyze GPU utilization
        if metrics.gpu_usage:
            if metrics.gpu_usage.get('avg_gpu_utilization', 0) < 50:
                issues["warning"].append("Low GPU utilization - GPUs may be underutilized")
                issues["recommendations"].append("Increase batch size or reduce worker count")
            
            if metrics.gpu_usage.get('avg_memory_utilization', 0) > 90:
                issues["critical"].append("High GPU memory usage - risk of OOM errors")
                issues["recommendations"].append("Reduce batch size or use gradient checkpointing")
        
        # Analyze resource utilization
        if metrics.resource_utilization.get('cpu_utilization_change', 0) < 0.1:
            issues["info"].append("Low CPU utilization - may be GPU-bound")
        
        if metrics.resource_utilization.get('gpu_utilization_change', 0) < 0.1:
            issues["warning"].append("Low GPU utilization - check GPU allocation and batch size")
        
        # Analyze pipeline statistics
        if metrics.pipeline_stats:
            for op in metrics.pipeline_stats.get('operators', []):
                if op.get('wall_time', 0) > metrics.total_time * 0.8:
                    issues["warning"].append(f"Operator '{op['name']}' is taking most of the time")
                    issues["recommendations"].append(f"Optimize operator '{op['name']}' or increase parallelism")
        
        return issues
    
    def generate_optimization_report(self, baseline_metrics: PerformanceMetrics, 
                                   optimized_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not baseline_metrics or not optimized_metrics:
            return {"error": "Both baseline and optimized metrics required"}
        
        # Calculate improvements
        time_improvement = baseline_metrics.total_time / optimized_metrics.total_time
        throughput_improvement = optimized_metrics.throughput / baseline_metrics.throughput
        memory_improvement = baseline_metrics.memory_usage.get('memory_increase_gb', 1) / max(optimized_metrics.memory_usage.get('memory_increase_gb', 1), 0.1)
        
        # Performance summary
        performance_summary = {
            "time_improvement": time_improvement,
            "throughput_improvement": throughput_improvement,
            "memory_efficiency_improvement": memory_improvement,
            "absolute_improvements": {
                "time_saved_seconds": baseline_metrics.total_time - optimized_metrics.total_time,
                "throughput_increase": optimized_metrics.throughput - baseline_metrics.throughput,
                "memory_saved_gb": baseline_metrics.memory_usage.get('memory_increase_gb', 0) - optimized_metrics.memory_usage.get('memory_increase_gb', 0)
            }
        }
        
        # Identify key optimizations that worked
        successful_optimizations = []
        
        if time_improvement > 2:
            successful_optimizations.append("Significant time reduction achieved")
        if throughput_improvement > 2:
            successful_optimizations.append("Major throughput improvement")
        if memory_improvement > 1.5:
            successful_optimizations.append("Better memory efficiency")
        
        # Generate recommendations
        recommendations = []
        
        if optimized_metrics.gpu_usage and optimized_metrics.gpu_usage.get('avg_gpu_utilization', 0) < 80:
            recommendations.append("Consider increasing batch size for better GPU utilization")
        
        if optimized_metrics.resource_utilization.get('cpu_utilization_change', 0) < 0.3:
            recommendations.append("CPU utilization is low - pipeline may be GPU-bound")
        
        if optimized_metrics.memory_usage.get('memory_increase_gb', 0) > 4:
            recommendations.append("Monitor memory usage - consider streaming processing for larger datasets")
        
        return {
            "performance_summary": performance_summary,
            "successful_optimizations": successful_optimizations,
            "recommendations": recommendations,
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": optimized_metrics
        }
    
    def save_analysis_report(self, report: Dict[str, Any], filename: str = "performance_analysis_report.json"):
        """Save analysis report to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Analysis report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
    
    def print_performance_summary(self, metrics: PerformanceMetrics, title: str):
        """Print formatted performance summary"""
        logger.info("\n" + "="*60)
        logger.info(f"📊 {title}")
        logger.info("="*60)
        logger.info(f"⏱️  Total Time: {metrics.total_time:.2f} seconds")
        logger.info(f"📊 Images Processed: {metrics.images_processed}")
        logger.info(f"🚀 Throughput: {metrics.throughput:.2f} images/second")
        logger.info(f"💾 Memory Increase: {metrics.memory_usage.get('memory_increase_gb', 0):.2f} GB")
        
        if metrics.gpu_usage:
            logger.info(f"🎮 GPU Count: {metrics.gpu_usage.get('gpu_count', 0)}")
            logger.info(f"🎮 Avg GPU Utilization: {metrics.gpu_usage.get('avg_gpu_utilization', 0):.1f}%")
            logger.info(f"🎮 Avg GPU Memory: {metrics.gpu_usage.get('avg_memory_utilization', 0):.1f}%")
        
        # Print issues if any
        issues = self.analyze_performance_issues(metrics)
        if issues["critical"]:
            logger.info("\n🚨 CRITICAL ISSUES:")
            for issue in issues["critical"]:
                logger.info(f"  • {issue}")
        
        if issues["warning"]:
            logger.info("\n⚠️  WARNINGS:")
            for issue in issues["warning"]:
                logger.info(f"  • {issue}")
        
        if issues["recommendations"]:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in issues["recommendations"]:
                logger.info(f"  • {rec}")

def run_comprehensive_analysis(baseline_pipeline_func, optimized_pipeline_func):
    """Run comprehensive performance analysis comparing baseline vs optimized"""
    analyzer = RayDataPerformanceAnalyzer()
    
    logger.info("🔍 Starting Comprehensive Performance Analysis")
    logger.info("This will run both pipelines and provide detailed comparison")
    
    # Run baseline performance test
    logger.info("\n" + "="*60)
    logger.info("🚀 RUNNING BASELINE PIPELINE")
    logger.info("="*60)
    
    baseline_metrics = analyzer.run_performance_test(baseline_pipeline_func, "Baseline Pipeline")
    analyzer.print_performance_summary(baseline_metrics, "BASELINE PERFORMANCE SUMMARY")
    
    # Run optimized performance test
    logger.info("\n" + "="*60)
    logger.info("✨ RUNNING OPTIMIZED PIPELINE")
    logger.info("="*60)
    
    optimized_metrics = analyzer.run_performance_test(optimized_pipeline_func, "Optimized Pipeline")
    analyzer.print_performance_summary(optimized_metrics, "OPTIMIZED PERFORMANCE SUMMARY")
    
    # Generate comparison report
    logger.info("\n" + "="*60)
    logger.info("📊 GENERATING COMPARISON REPORT")
    logger.info("="*60)
    
    comparison_report = analyzer.generate_optimization_report(baseline_metrics, optimized_metrics)
    
    # Print comparison summary
    logger.info("\n🎯 OPTIMIZATION RESULTS:")
    logger.info(f"⏱️  Time Improvement: {comparison_report['performance_summary']['time_improvement']:.2f}x faster")
    logger.info(f"🚀 Throughput Improvement: {comparison_report['performance_summary']['throughput_improvement']:.2f}x higher")
    logger.info(f"💾 Memory Efficiency: {comparison_report['performance_summary']['memory_efficiency_improvement']:.2f}x better")
    
    if comparison_report['successful_optimizations']:
        logger.info("\n✅ SUCCESSFUL OPTIMIZATIONS:")
        for opt in comparison_report['successful_optimizations']:
            logger.info(f"  • {opt}")
    
    if comparison_report['recommendations']:
        logger.info("\n💡 FURTHER OPTIMIZATION RECOMMENDATIONS:")
        for rec in comparison_report['recommendations']:
            logger.info(f"  • {rec}")
    
    # Save detailed report
    analyzer.save_analysis_report(comparison_report)
    
    return comparison_report

if __name__ == "__main__":
    logger.info("Performance Analysis Tools for Ray Data Pipelines")
    logger.info("Import and use these tools to analyze your pipeline performance")
