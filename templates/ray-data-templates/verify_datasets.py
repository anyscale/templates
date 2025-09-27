#!/usr/bin/env python3
"""
Dataset Verification Script

Verifies that all datasets required by Ray Data templates are accessible
from the s3://ray-benchmark-data bucket.

Usage:
    python verify_datasets.py
"""

import ray
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_dataset(path: str, description: str, expected_min_records: int = 1) -> Tuple[bool, str, int]:
    """Verify a single dataset is accessible and has expected data."""
    try:
        if path.endswith('.parquet'):
            dataset = ray.data.read_parquet(path)
        elif path.endswith('.csv'):
            dataset = ray.data.read_csv(path)
        elif path.endswith('.json'):
            dataset = ray.data.read_json(path)
        elif path.endswith('.txt') or path.endswith('.log'):
            dataset = ray.data.read_text(path)
        elif '/train/' in path or '/images/' in path:
            dataset = ray.data.read_images(path)
        else:
            return False, f"Unknown file format for {path}", 0
        
        count = dataset.count()
        
        if count >= expected_min_records:
            return True, f"PASS {description}: {count:,} records", count
        else:
            return False, f"FAIL {description}: Only {count} records (expected >= {expected_min_records})", count
            
    except Exception as e:
        return False, f"ERROR {description}: {str(e)}", 0

def main():
    """Main verification function."""
    logger.info("Starting Ray Data template dataset verification...")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define all datasets to verify
    datasets = [
        # Financial data
        ("s3://ray-benchmark-data/financial/sp500_daily_2years.parquet", 
         "Financial S&P 500 data", 1000),
        
        # ML datasets
        ("s3://ray-benchmark-data/ml-datasets/titanic.csv", 
         "Titanic ML dataset", 800),
        
        # Geospatial data
        ("s3://ray-benchmark-data/nyc-taxi/yellow_tripdata_2023-01.parquet", 
         "NYC Taxi geospatial data", 10000),
        
        # TPC-H benchmark data
        ("s3://ray-benchmark-data/tpch/parquet/sf10/customer.parquet", 
         "TPC-H Customer data", 100000),
        ("s3://ray-benchmark-data/tpch/parquet/sf10/nation.parquet", 
         "TPC-H Nation reference data", 25),
        ("s3://ray-benchmark-data/tpch/parquet/sf10/orders.parquet", 
         "TPC-H Orders data", 500000),
        ("s3://ray-benchmark-data/tpch/parquet/sf10/lineitem.parquet", 
         "TPC-H LineItem data", 1000000),
        
        # Log data
        ("s3://ray-benchmark-data/logs/apache-access.log", 
         "Apache access logs", 100),
        ("s3://ray-benchmark-data/logs/application.json", 
         "Application JSON logs", 100),
        ("s3://ray-benchmark-data/logs/security.log", 
         "Security text logs", 100),
        
        # Medical data
        ("s3://ray-benchmark-data/medical/dicom-metadata.json", 
         "DICOM medical metadata", 10),
        ("s3://ray-benchmark-data/medical/hl7-messages/messages.hl7", 
         "HL7 medical messages", 10),
        ("s3://ray-benchmark-data/medical/patient-records.csv", 
         "Patient medical records", 100),
        ("s3://ray-benchmark-data/medical/laboratory_results.parquet", 
         "Laboratory results", 100),
        
        # Text data
        ("s3://ray-benchmark-data/text/captions.txt", 
         "Text captions for multimodal AI", 1000),
        
        # Support data
        ("s3://ray-benchmark-data/support/tickets/tickets.json", 
         "Support tickets for NLP", 1000),
        
        # Data quality/catalog data
        ("s3://ray-benchmark-data/catalog/customer_data.parquet", 
         "Customer data for quality monitoring", 1000),
    ]
    
    # Verify each dataset
    results = []
    total_datasets = len(datasets)
    successful_datasets = 0
    
    logger.info(f"Verifying {total_datasets} datasets...")
    print("\n" + "="*80)
    print("RAY DATA TEMPLATE DATASET VERIFICATION")
    print("="*80)
    
    for path, description, min_records in datasets:
        success, message, count = verify_dataset(path, description, min_records)
        results.append((success, message, count, path))
        
        if success:
            successful_datasets += 1
            
        print(message)
    
    # Image datasets (special handling)
    print("\n" + "-"*80)
    print("IMAGE DATASETS (require manual ImageNette download)")
    print("-"*80)
    
    image_datasets = [
        ("s3://ray-benchmark-data/imagenette2/train/", 
         "ImageNette training images", 10),
    ]
    
    for path, description, min_records in image_datasets:
        success, message, count = verify_dataset(path, description, min_records)
        results.append((success, message, count, path))
        print(message)
        if success:
            successful_datasets += 1
    
    # Summary
    total_with_images = total_datasets + len(image_datasets)
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Datasets verified: {successful_datasets}/{total_with_images}")
    print(f"Success rate: {(successful_datasets/total_with_images)*100:.1f}%")
    
    if successful_datasets == total_with_images:
        print("\n🎉 All datasets verified successfully!")
        print("Ray Data templates are ready to use.")
    elif successful_datasets >= total_datasets:
        print("\n✅ Core datasets verified successfully!")
        print("📝 Note: ImageNette requires manual download (see DATASET_UPLOAD_README.md)")
        print("Ray Data templates are ready to use (except image-based templates).")
    else:
        print(f"\n⚠️  {total_with_images - successful_datasets} datasets failed verification.")
        print("Some Ray Data templates may not work correctly.")
        print("\nFailed datasets:")
        for success, message, count, path in results:
            if not success:
                print(f"  - {path}")
        print("\nRun 'python upload_benchmark_data.py' to upload missing datasets.")
    
    # Template readiness
    print("\n" + "-"*80)
    print("TEMPLATE READINESS")
    print("-"*80)
    
    template_datasets = {
        "ray-data-financial-forecasting": ["financial/sp500_daily_2years.parquet"],
        "ray-data-ml-feature-engineering": ["ml-datasets/titanic.csv"],
        "ray-data-geospatial-analysis": ["nyc-taxi/yellow_tripdata_2023-01.parquet"],
        "ray-data-etl-tpch": ["tpch/parquet/sf10/customer.parquet", "tpch/parquet/sf10/nation.parquet", 
                              "tpch/parquet/sf10/orders.parquet", "tpch/parquet/sf10/lineitem.parquet"],
        "ray-data-log-ingestion": ["logs/apache-access.log", "logs/application.json", "logs/security.log"],
        "ray-data-medical-connectors": ["medical/dicom-metadata.json", "medical/hl7-messages/messages.hl7", 
                                       "medical/patient-records.csv", "medical/laboratory_results.parquet"],
        "ray-data-nlp-text-analytics": ["support/tickets/tickets.json"],
        "ray-data-multimodal-ai-pipeline": ["imagenette2/train/", "text/captions.txt"],
        "ray-data-batch-inference-optimization": ["imagenette2/train/"],
        "ray-data-data-quality-monitoring": ["catalog/customer_data.parquet"],
        "ray-data-enterprise-data-catalog": ["catalog/customer_data.parquet"],
    }
    
    verified_paths = {result[3].replace("s3://ray-benchmark-data/", "") for result in results if result[0]}
    
    for template, required_paths in template_datasets.items():
        missing = [path for path in required_paths if path not in verified_paths]
        if not missing:
            print(f"✅ {template}: Ready")
        else:
            print(f"❌ {template}: Missing {len(missing)} datasets")
            for path in missing:
                print(f"   - {path}")
    
    ray.shutdown()
    
    return 0 if successful_datasets >= total_datasets else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
