#!/usr/bin/env python3
"""
TPC-H ETL Pipeline Demo with Ray Data

This script demonstrates a comprehensive ETL pipeline using TPC-H benchmark data
and Ray Data for distributed processing. The pipeline includes data extraction,
transformation, aggregation, and loading with performance optimization.

Key Features:
- TPC-H benchmark data generation and processing
- SQL-like transformations and business logic
- Performance benchmarking and optimization
- Production-ready error handling and monitoring
"""

import os
import logging
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import tempfile

import ray
from ray.data import Dataset
from ray.data.context import DataContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable progress bars for cleaner output
DataContext.get_current().enable_progress_bars = False


class TPCHDataGenerator:
    """Generate TPC-H benchmark data for ETL demonstration."""
    
    def __init__(self, scale_factor: float = 0.1):
        """Initialize TPC-H data generator."""
        self.scale_factor = scale_factor
        self.num_customers = int(1000 * scale_factor)
        self.num_orders = int(5000 * scale_factor)
        self.num_lineitems = int(20000 * scale_factor)
        self.num_parts = int(2000 * scale_factor)
        self.num_suppliers = int(100 * scale_factor)
        
        logger.info(f"Initialized TPC-H generator with scale factor {scale_factor}")
    
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer table data."""
        logger.info(f"Generating {self.num_customers} customers...")
        
        customers = pd.DataFrame({
            'c_custkey': range(1, self.num_customers + 1),
            'c_name': [f'Customer_{i:06d}' for i in range(1, self.num_customers + 1)],
            'c_address': [f'{np.random.randint(1, 9999)} Main St, City {i%100}' for i in range(self.num_customers)],
            'c_nationkey': np.random.randint(0, 25, self.num_customers),
            'c_phone': [f'{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' 
                       for _ in range(self.num_customers)],
            'c_acctbal': np.random.uniform(-999.99, 9999.99, self.num_customers),
            'c_mktsegment': np.random.choice(['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'], 
                                           self.num_customers),
            'c_comment': [f'Customer comment {i}' for i in range(self.num_customers)]
        })
        
        return customers
    
    def generate_orders(self) -> pd.DataFrame:
        """Generate orders table data."""
        logger.info(f"Generating {self.num_orders} orders...")
        
        # Generate order dates over the last 7 years
        start_date = pd.Timestamp('2017-01-01')
        end_date = pd.Timestamp('2024-01-01')
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        orders = pd.DataFrame({
            'o_orderkey': range(1, self.num_orders + 1),
            'o_custkey': np.random.randint(1, self.num_customers + 1, self.num_orders),
            'o_orderstatus': np.random.choice(['O', 'F', 'P'], self.num_orders, p=[0.5, 0.4, 0.1]),
            'o_totalprice': np.random.uniform(100.00, 50000.00, self.num_orders),
            'o_orderdate': np.random.choice(date_range, self.num_orders),
            'o_orderpriority': np.random.choice(['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW'], 
                                              self.num_orders),
            'o_clerk': [f'Clerk#{np.random.randint(1, 1000):04d}' for _ in range(self.num_orders)],
            'o_shippriority': np.random.randint(0, 2, self.num_orders),
            'o_comment': [f'Order comment {i}' for i in range(self.num_orders)]
        })
        
        return orders
    
    def generate_lineitems(self) -> pd.DataFrame:
        """Generate lineitem table data."""
        logger.info(f"Generating {self.num_lineitems} line items...")
        
        lineitems = pd.DataFrame({
            'l_orderkey': np.random.randint(1, self.num_orders + 1, self.num_lineitems),
            'l_partkey': np.random.randint(1, self.num_parts + 1, self.num_lineitems),
            'l_suppkey': np.random.randint(1, self.num_suppliers + 1, self.num_lineitems),
            'l_linenumber': [i % 7 + 1 for i in range(self.num_lineitems)],
            'l_quantity': np.random.randint(1, 51, self.num_lineitems),
            'l_extendedprice': np.random.uniform(100.00, 10000.00, self.num_lineitems),
            'l_discount': np.random.uniform(0.00, 0.10, self.num_lineitems),
            'l_tax': np.random.uniform(0.00, 0.08, self.num_lineitems),
            'l_returnflag': np.random.choice(['A', 'N', 'R'], self.num_lineitems),
            'l_linestatus': np.random.choice(['O', 'F'], self.num_lineitems),
            'l_shipdate': pd.date_range('2017-01-01', periods=self.num_lineitems, freq='D')[:self.num_lineitems],
            'l_commitdate': pd.date_range('2017-01-01', periods=self.num_lineitems, freq='D')[:self.num_lineitems],
            'l_receiptdate': pd.date_range('2017-01-01', periods=self.num_lineitems, freq='D')[:self.num_lineitems],
            'l_shipinstruct': np.random.choice(['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN'], 
                                             self.num_lineitems),
            'l_shipmode': np.random.choice(['TRUCK', 'MAIL', 'SHIP', 'AIR', 'REG AIR', 'RAIL', 'FOB'], 
                                         self.num_lineitems),
            'l_comment': [f'Line item comment {i}' for i in range(self.num_lineitems)]
        })
        
        return lineitems


class TPCHETLProcessor:
    """Process TPC-H data through comprehensive ETL pipeline."""
    
    def __init__(self):
        """Initialize ETL processor."""
        self.processing_stats = {}
        logger.info("Initialized TPC-H ETL processor")
    
    def extract_data(self, scale_factor: float = 0.1) -> tuple:
        """Extract TPC-H data and create Ray datasets."""
        logger.info("Starting data extraction phase...")
        start_time = time.time()
        
        # Generate TPC-H data
        generator = TPCHDataGenerator(scale_factor)
        
        customers_df = generator.generate_customers()
        orders_df = generator.generate_orders()
        lineitems_df = generator.generate_lineitems()
        
        # Convert to Ray datasets
        customers_ds = ray.data.from_pandas(customers_df)
        orders_ds = ray.data.from_pandas(orders_df)
        lineitems_ds = ray.data.from_pandas(lineitems_df)
        
        extraction_time = time.time() - start_time
        total_records = customers_ds.count() + orders_ds.count() + lineitems_ds.count()
        
        logger.info(f"Data extraction completed in {extraction_time:.2f}s")
        logger.info(f"Total records extracted: {total_records:,}")
        
        self.processing_stats['extraction'] = {
            'time': extraction_time,
            'records': total_records,
            'throughput': total_records / extraction_time if extraction_time > 0 else 0
        }
        
        return customers_ds, orders_ds, lineitems_ds
    
    def transform_customers(self, customers_ds: Dataset) -> Dataset:
        """Transform customer data with business logic."""
        logger.info("Transforming customer data...")
        start_time = time.time()
        
        def customer_transformation(batch):
            """Apply customer-specific transformations."""
            import pandas as pd
            
            # Convert batch to DataFrame for easier processing
            df = pd.DataFrame(batch)
            
            # Calculate customer segment based on account balance
            df['customer_segment'] = pd.cut(
                df['c_acctbal'], 
                bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                labels=['Negative', 'Basic', 'Standard', 'Premium']
            )
            
            # Calculate value tier
            df['value_tier'] = pd.cut(
                df['c_acctbal'],
                bins=[-float('inf'), 2000, 5000, 8000, float('inf')],
                labels=['Bronze', 'Silver', 'Gold', 'Platinum']
            )
            
            # Add derived fields
            df['account_balance_tier'] = (df['c_acctbal'] // 1000).astype(int)
            df['is_high_value'] = df['c_acctbal'] > 7500
            df['is_negative_balance'] = df['c_acctbal'] < 0
            df['phone_area_code'] = df['c_phone'].str[:3]
            df['market_segment_code'] = df['c_mktsegment'].str[:3].str.upper()
            df['processing_timestamp'] = pd.Timestamp.now().isoformat()
            
            return {"transformed_customers": df.to_dict('records')}
        
        # Apply transformations
        transformed_ds = customers_ds.map_batches(
            customer_transformation,
            batch_size=1000,
            concurrency=4
        )
        
        transformation_time = time.time() - start_time
        record_count = transformed_ds.count()
        
        logger.info(f"Customer transformation completed in {transformation_time:.2f}s")
        
        self.processing_stats['customer_transform'] = {
            'time': transformation_time,
            'records': record_count,
            'throughput': record_count / transformation_time if transformation_time > 0 else 0
        }
        
        return transformed_ds
    
    def transform_orders(self, orders_ds: Dataset) -> Dataset:
        """Transform orders data with business logic."""
        logger.info("Transforming orders data...")
        start_time = time.time()
        
        def order_transformation(batch):
            """Apply order-specific transformations."""
            transformed_orders = []
            
            for order in batch:
                # Extract date components
                order_date = pd.to_datetime(order['o_orderdate'])
                
                # Calculate order metrics
                total_price = order['o_totalprice']
                
                # Determine order size category
                if total_price > 30000:
                    order_size = 'Large'
                elif total_price > 10000:
                    order_size = 'Medium'
                else:
                    order_size = 'Small'
                
                # Calculate business metrics
                priority_score = {
                    '1-URGENT': 5,
                    '2-HIGH': 4,
                    '3-MEDIUM': 3,
                    '4-NOT SPECIFIED': 2,
                    '5-LOW': 1
                }.get(order['o_orderpriority'], 0)
                
                # Add derived fields
                transformed_order = {
                    **order,
                    'order_year': order_date.year,
                    'order_month': order_date.month,
                    'order_quarter': order_date.quarter,
                    'order_day_of_week': order_date.dayofweek,
                    'order_size': order_size,
                    'priority_score': priority_score,
                    'is_urgent': order['o_orderpriority'] in ['1-URGENT', '2-HIGH'],
                    'is_weekend_order': order_date.dayofweek >= 5,
                    'days_since_epoch': (order_date - pd.Timestamp('2000-01-01')).days,
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
                
                transformed_orders.append(transformed_order)
            
            return transformed_orders
        
        # Apply transformations
        transformed_ds = orders_ds.map_batches(
            order_transformation,
            batch_size=1000,
            concurrency=4
        )
        
        transformation_time = time.time() - start_time
        record_count = transformed_ds.count()
        
        logger.info(f"Orders transformation completed in {transformation_time:.2f}s")
        
        self.processing_stats['orders_transform'] = {
            'time': transformation_time,
            'records': record_count,
            'throughput': record_count / transformation_time if transformation_time > 0 else 0
        }
        
        return transformed_ds
    
    def perform_business_aggregations(self, customers_ds: Dataset, orders_ds: Dataset) -> Dataset:
        """Perform business intelligence aggregations."""
        logger.info("Performing business aggregations...")
        start_time = time.time()
        
        def calculate_customer_metrics(batch):
            """Calculate customer-level business metrics."""
            # This would typically involve joining with orders data
            # For demonstration, we'll calculate metrics from customer data
            
            df = pd.DataFrame(batch)
            
            # Calculate segment-level metrics
            segment_metrics = df.groupby('customer_segment').agg({
                'c_acctbal': ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'c_custkey': 'count',
                'is_high_value': 'sum'
            }).round(2)
            
            # Flatten column names
            segment_metrics.columns = ['_'.join(col).strip() for col in segment_metrics.columns]
            segment_metrics = segment_metrics.reset_index()
            
            # Add derived metrics
            segment_metrics['avg_account_balance'] = segment_metrics['c_acctbal_sum'] / segment_metrics['c_acctbal_count']
            segment_metrics['high_value_customer_rate'] = segment_metrics['is_high_value_sum'] / segment_metrics['c_custkey_count']
            segment_metrics['balance_coefficient_variation'] = segment_metrics['c_acctbal_std'] / segment_metrics['c_acctbal_mean']
            
            # Add metadata
            segment_metrics['calculation_timestamp'] = pd.Timestamp.now().isoformat()
            segment_metrics['data_source'] = 'tpch_customers'
            segment_metrics['scale_factor'] = self.scale_factor
            
            return segment_metrics.to_dict('records')
        
        # Apply aggregations
        aggregated_ds = customers_ds.map_batches(
            calculate_customer_metrics,
            batch_size=2000,
            concurrency=2
        )
        
        aggregation_time = time.time() - start_time
        record_count = aggregated_ds.count()
        
        logger.info(f"Business aggregations completed in {aggregation_time:.2f}s")
        
        self.processing_stats['aggregation'] = {
            'time': aggregation_time,
            'records': record_count,
            'throughput': record_count / aggregation_time if aggregation_time > 0 else 0
        }
        
        return aggregated_ds
    
    def load_results(self, datasets: Dict[str, Dataset], output_dir: str):
        """Load results to multiple destinations."""
        logger.info("Starting data loading phase...")
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        total_records = 0
        
        for name, dataset in datasets.items():
            try:
                # Write to Parquet (efficient for analytics)
                parquet_path = f"local://{output_dir}/{name}_parquet"
                dataset.write_parquet(parquet_path)
                logger.info(f"Saved {name} to Parquet format")
                
                # Write to CSV (human-readable)
                csv_path = f"local://{output_dir}/{name}_csv"
                dataset.write_csv(csv_path)
                logger.info(f"Saved {name} to CSV format")
                
                # Count records
                record_count = dataset.count()
                total_records += record_count
                logger.info(f"{name}: {record_count:,} records")
                
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
                continue
        
        loading_time = time.time() - start_time
        
        logger.info(f"Data loading completed in {loading_time:.2f}s")
        logger.info(f"Total records loaded: {total_records:,}")
        
        self.processing_stats['loading'] = {
            'time': loading_time,
            'records': total_records,
            'throughput': total_records / loading_time if loading_time > 0 else 0
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.processing_stats:
            return "No performance data available"
        
        report = "\n" + "="*70 + "\n"
        report += "TPC-H ETL PIPELINE PERFORMANCE REPORT\n"
        report += "="*70 + "\n"
        
        # Overall statistics
        total_time = sum(stage['time'] for stage in self.processing_stats.values())
        total_records = sum(stage['records'] for stage in self.processing_stats.values())
        overall_throughput = total_records / total_time if total_time > 0 else 0
        
        report += f"Overall Performance:\n"
        report += f"  Scale Factor: {self.scale_factor}\n"
        report += f"  Total Records: {total_records:,}\n"
        report += f"  Total Time: {total_time:.2f}s\n"
        report += f"  Overall Throughput: {overall_throughput:,.2f} records/sec\n\n"
        
        # Stage-by-stage performance
        report += "Stage Performance:\n"
        for stage_name, stats in self.processing_stats.items():
            report += f"  {stage_name.title()}:\n"
            report += f"    Records: {stats['records']:,}\n"
            report += f"    Time: {stats['time']:.2f}s\n"
            report += f"    Throughput: {stats['throughput']:,.2f} records/sec\n"
        
        report += "\n"
        
        # Performance recommendations
        report += "Performance Recommendations:\n"
        slowest_stage = min(self.processing_stats.items(), key=lambda x: x[1]['throughput'])
        fastest_stage = max(self.processing_stats.items(), key=lambda x: x[1]['throughput'])
        
        report += f"  Fastest Stage: {fastest_stage[0]} ({fastest_stage[1]['throughput']:,.2f} records/sec)\n"
        report += f"  Slowest Stage: {slowest_stage[0]} ({slowest_stage[1]['throughput']:,.2f} records/sec)\n"
        report += f"  Optimization Focus: {slowest_stage[0]}\n"
        
        return report


def run_tpch_etl_pipeline(scale_factor: float = 0.1):
    """Run the complete TPC-H ETL pipeline."""
    logger.info("Starting TPC-H ETL pipeline...")
    pipeline_start_time = time.time()
    
    # Initialize ETL processor
    processor = TPCHETLProcessor()
    
    try:
        # Phase 1: Extract
        customers_ds, orders_ds, lineitems_ds = processor.extract_data(scale_factor)
        
        # Phase 2: Transform
        transformed_customers = processor.transform_customers(customers_ds)
        transformed_orders = processor.transform_orders(orders_ds)
        
        # Phase 3: Aggregate
        business_metrics = processor.perform_business_aggregations(transformed_customers, transformed_orders)
        
        # Phase 4: Load
        output_dir = tempfile.mkdtemp()
        datasets_to_save = {
            'transformed_customers': transformed_customers,
            'transformed_orders': transformed_orders,
            'business_metrics': business_metrics
        }
        
        processor.load_results(datasets_to_save, output_dir)
        
        # Generate performance report
        performance_report = processor.generate_performance_report()
        logger.info(performance_report)
        
        # Save performance report
        with open(f"{output_dir}/performance_report.txt", "w") as f:
            f.write(performance_report)
        
        pipeline_time = time.time() - pipeline_start_time
        logger.info(f"Complete ETL pipeline finished in {pipeline_time:.2f}s")
        
        return {
            'output_dir': output_dir,
            'performance_stats': processor.processing_stats,
            'pipeline_time': pipeline_time
        }
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


def benchmark_etl_performance():
    """Benchmark ETL performance across different configurations."""
    logger.info("Starting ETL performance benchmark...")
    
    scale_factors = [0.05, 0.1, 0.2]  # Different data sizes
    benchmark_results = []
    
    for scale_factor in scale_factors:
        logger.info(f"Benchmarking with scale factor {scale_factor}")
        
        try:
            # Run ETL pipeline
            result = run_tpch_etl_pipeline(scale_factor)
            
            benchmark_result = {
                'scale_factor': scale_factor,
                'pipeline_time': result['pipeline_time'],
                'performance_stats': result['performance_stats']
            }
            
            benchmark_results.append(benchmark_result)
            
        except Exception as e:
            logger.error(f"Benchmark failed for scale factor {scale_factor}: {e}")
            continue
    
    # Generate benchmark report
    if benchmark_results:
        logger.info("\n" + "="*60)
        logger.info("ETL PERFORMANCE BENCHMARK RESULTS")
        logger.info("="*60)
        
        for result in benchmark_results:
            scale = result['scale_factor']
            time_taken = result['pipeline_time']
            extraction_throughput = result['performance_stats']['extraction']['throughput']
            
            logger.info(f"Scale Factor {scale}:")
            logger.info(f"  Pipeline Time: {time_taken:.2f}s")
            logger.info(f"  Extraction Throughput: {extraction_throughput:,.2f} records/sec")
            logger.info("")
    
    return benchmark_results


def main():
    """Main function to run TPC-H ETL demo."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
    
    try:
        # Run standard ETL pipeline
        logger.info("Running standard TPC-H ETL pipeline...")
        result = run_tpch_etl_pipeline(scale_factor=0.1)
        
        # Run performance benchmark
        logger.info("Running ETL performance benchmark...")
        benchmark_results = benchmark_etl_performance()
        
        # Display final summary
        logger.info("\n" + "="*50)
        logger.info("TPC-H ETL DEMO SUMMARY")
        logger.info("="*50)
        logger.info(f"Pipeline completed successfully")
        logger.info(f"Results saved to: {result['output_dir']}")
        logger.info(f"Benchmark configurations tested: {len(benchmark_results)}")
        
    except Exception as e:
        logger.error(f"TPC-H ETL demo failed: {e}")
        raise
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
