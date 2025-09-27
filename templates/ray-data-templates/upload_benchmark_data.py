#!/usr/bin/env python3
"""
Ray Data Template Benchmark Data Upload Script

This script uploads all required datasets to the s3://ray-benchmark-data bucket
to support the Ray Data templates. It handles both local files and external
datasets that need to be downloaded and processed.

Usage:
    python upload_benchmark_data.py

Requirements:
    pip install boto3 pandas pyarrow requests yfinance
    AWS credentials configured for s3://ray-benchmark-data bucket
"""

import os
import boto3
import pandas as pd
import numpy as np
import requests
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# S3 Configuration
BUCKET_NAME = "ray-benchmark-data"
BASE_PATH = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"

class BenchmarkDataUploader:
    """Handles uploading benchmark datasets to S3 for Ray Data templates."""
    
    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized uploader for bucket: {bucket_name}")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def upload_file_to_s3(self, local_path: str, s3_key: str) -> bool:
        """Upload a file to S3."""
        try:
            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def upload_existing_parquet_files(self):
        """Upload existing parquet files from template directories."""
        logger.info("Starting upload of existing parquet files...")
        
        # Medical data files
        medical_files = [
            ("ray-data-medical-connectors/laboratory_results.parquet", "medical/laboratory_results.parquet"),
            ("ray-data-medical-connectors/patient_medical_records.parquet", "medical/patient-records.csv"),  # Convert to CSV as expected
            ("ray-data-medical-connectors/dicom_imaging_metadata.parquet", "medical/dicom-metadata.json"),  # Convert to JSON as expected
            ("ray-data-medical-connectors/hl7_medical_messages.parquet", "medical/hl7-messages/messages.hl7"),  # Convert to HL7 text format
        ]
        
        # Log data files
        log_files = [
            ("ray-data-log-ingestion/apache_access_logs.parquet", "logs/apache-access.log"),  # Convert to text format
            ("ray-data-log-ingestion/application_logs.parquet", "logs/application.json"),  # Convert to JSON format
            ("ray-data-log-ingestion/security_logs.parquet", "logs/security.log"),  # Convert to text format
        ]
        
        # Data quality files
        quality_files = [
            ("ray-data-data-quality-monitoring/ecommerce_customers_with_quality_issues.parquet", "catalog/customer_data.parquet"),
        ]
        
        all_files = medical_files + log_files + quality_files
        
        for local_rel_path, s3_key in all_files:
            local_path = os.path.join(BASE_PATH, local_rel_path)
            if os.path.exists(local_path):
                # For files that need format conversion, handle appropriately
                if s3_key.endswith('.csv') or s3_key.endswith('.json') or s3_key.endswith('.log') or s3_key.endswith('.hl7'):
                    self._convert_and_upload_parquet(local_path, s3_key)
                else:
                    self.upload_file_to_s3(local_path, s3_key)
            else:
                logger.warning(f"Local file not found: {local_path}")
    
    def _convert_and_upload_parquet(self, parquet_path: str, s3_key: str):
        """Convert parquet files to required formats and upload."""
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            temp_path = os.path.join(self.temp_dir, os.path.basename(s3_key))
            
            if s3_key.endswith('.csv'):
                df.to_csv(temp_path, index=False)
            elif s3_key.endswith('.json'):
                df.to_json(temp_path, orient='records', lines=True)
            elif s3_key.endswith('.log') or s3_key.endswith('.hl7'):
                # Convert to text format (assuming there's a 'message' or 'log' column)
                if 'message' in df.columns:
                    with open(temp_path, 'w') as f:
                        for message in df['message']:
                            f.write(str(message) + '\n')
                elif 'log' in df.columns:
                    with open(temp_path, 'w') as f:
                        for log in df['log']:
                            f.write(str(log) + '\n')
                else:
                    # Use first string column
                    string_cols = df.select_dtypes(include=['object']).columns
                    if len(string_cols) > 0:
                        with open(temp_path, 'w') as f:
                            for value in df[string_cols[0]]:
                                f.write(str(value) + '\n')
            
            self.upload_file_to_s3(temp_path, s3_key)
            os.remove(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to convert and upload {parquet_path}: {e}")
    
    def create_financial_data(self):
        """Create S&P 500 financial data."""
        logger.info("Creating S&P 500 financial dataset...")
        
        try:
            import yfinance as yf
            
            # Major S&P 500 symbols
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
            
            # Get 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            all_data = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    for date, row in hist.iterrows():
                        record = {
                            'symbol': symbol,
                            'date': date.strftime('%Y-%m-%d'),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume'])
                        }
                        all_data.append(record)
                        
                    logger.info(f"Downloaded data for {symbol}: {len(hist)} days")
                    
                except Exception as e:
                    logger.error(f"Failed to download {symbol}: {e}")
            
            # Save to parquet
            df = pd.DataFrame(all_data)
            temp_path = os.path.join(self.temp_dir, "sp500_daily_2years.parquet")
            df.to_parquet(temp_path)
            
            self.upload_file_to_s3(temp_path, "financial/sp500_daily_2years.parquet")
            os.remove(temp_path)
            
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            # Create synthetic financial data
            self._create_synthetic_financial_data()
        except Exception as e:
            logger.error(f"Failed to create financial data: {e}")
            self._create_synthetic_financial_data()
    
    def _create_synthetic_financial_data(self):
        """Create synthetic financial data as fallback."""
        logger.info("Creating synthetic financial data...")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        base_prices = [150, 300, 2500, 3000, 200, 250, 400, 140, 160, 200]
        
        all_data = []
        end_date = datetime.now()
        
        for i, symbol in enumerate(symbols):
            current_price = base_prices[i]
            
            for days_back in range(730):
                date = end_date - timedelta(days=days_back)
                
                # Simple random walk
                change = np.random.normal(0, 0.02)
                current_price *= (1 + change)
                
                # Ensure positive prices
                current_price = max(current_price, 1.0)
                
                volume = int(np.random.normal(1000000, 300000))
                volume = max(volume, 100000)
                
                record = {
                    'symbol': symbol,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(current_price * np.random.uniform(0.99, 1.01), 2),
                    'high': round(current_price * np.random.uniform(1.00, 1.05), 2),
                    'low': round(current_price * np.random.uniform(0.95, 1.00), 2),
                    'close': round(current_price, 2),
                    'volume': volume
                }
                all_data.append(record)
        
        df = pd.DataFrame(all_data)
        temp_path = os.path.join(self.temp_dir, "sp500_daily_2years.parquet")
        df.to_parquet(temp_path)
        
        self.upload_file_to_s3(temp_path, "financial/sp500_daily_2years.parquet")
        os.remove(temp_path)
    
    def create_titanic_dataset(self):
        """Create Titanic dataset for ML feature engineering."""
        logger.info("Creating Titanic dataset...")
        
        try:
            # Try to download real Titanic dataset
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            response = requests.get(url)
            
            if response.status_code == 200:
                temp_path = os.path.join(self.temp_dir, "titanic.csv")
                with open(temp_path, 'w') as f:
                    f.write(response.text)
                
                self.upload_file_to_s3(temp_path, "ml-datasets/titanic.csv")
                os.remove(temp_path)
                logger.info("Successfully downloaded and uploaded real Titanic dataset")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to download Titanic dataset: {e}")
            self._create_synthetic_titanic_data()
    
    def _create_synthetic_titanic_data(self):
        """Create synthetic Titanic-like dataset."""
        logger.info("Creating synthetic Titanic dataset...")
        
        np.random.seed(42)
        n_passengers = 5000
        
        data = {
            'PassengerId': range(1, n_passengers + 1),
            'Survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.3, 0.2, 0.5]),
            'Name': [f"Passenger_{i}" for i in range(1, n_passengers + 1)],
            'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 12, n_passengers).clip(0, 80),
            'SibSp': np.random.poisson(0.5, n_passengers),
            'Parch': np.random.poisson(0.4, n_passengers),
            'Ticket': [f"TICKET_{i}" for i in range(1, n_passengers + 1)],
            'Fare': np.random.lognormal(3, 1, n_passengers),
            'Cabin': [f"C{i}" if np.random.random() > 0.7 else None for i in range(n_passengers)],
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.7, 0.2, 0.1])
        }
        
        df = pd.DataFrame(data)
        temp_path = os.path.join(self.temp_dir, "titanic.csv")
        df.to_csv(temp_path, index=False)
        
        self.upload_file_to_s3(temp_path, "ml-datasets/titanic.csv")
        os.remove(temp_path)
    
    def create_nyc_taxi_data(self):
        """Create NYC taxi data for geospatial analysis."""
        logger.info("Creating NYC taxi dataset...")
        
        # NYC boundaries (approximate)
        min_lat, max_lat = 40.4774, 40.9176
        min_lon, max_lon = -74.2591, -73.7004
        
        n_trips = 250000
        np.random.seed(42)
        
        data = []
        for i in range(n_trips):
            pickup_lat = np.random.uniform(min_lat, max_lat)
            pickup_lon = np.random.uniform(min_lon, max_lon)
            
            # Dropoff within reasonable distance
            dropoff_lat = pickup_lat + np.random.normal(0, 0.01)
            dropoff_lon = pickup_lon + np.random.normal(0, 0.01)
            
            trip = {
                'pickup_latitude': pickup_lat,
                'pickup_longitude': pickup_lon,
                'dropoff_latitude': dropoff_lat,
                'dropoff_longitude': dropoff_lon,
                'passenger_count': np.random.choice([1, 2, 3, 4, 5], p=[0.7, 0.15, 0.08, 0.05, 0.02]),
                'trip_distance': np.random.lognormal(1, 0.5),
                'fare_amount': np.random.lognormal(2.5, 0.5),
                'pickup_datetime': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat()
            }
            data.append(trip)
        
        df = pd.DataFrame(data)
        temp_path = os.path.join(self.temp_dir, "yellow_tripdata_2023-01.parquet")
        df.to_parquet(temp_path)
        
        self.upload_file_to_s3(temp_path, "nyc-taxi/yellow_tripdata_2023-01.parquet")
        os.remove(temp_path)
    
    def create_tpch_data(self):
        """Create TPC-H benchmark data."""
        logger.info("Creating TPC-H benchmark datasets...")
        
        # Create customer table
        n_customers = 500000
        customers = {
            'c_custkey': range(1, n_customers + 1),
            'c_name': [f"Customer#{i:09d}" for i in range(1, n_customers + 1)],
            'c_address': [f"Address {i}" for i in range(1, n_customers + 1)],
            'c_nationkey': np.random.randint(0, 25, n_customers),
            'c_phone': [f"{np.random.randint(10, 99)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" 
                       for _ in range(n_customers)],
            'c_acctbal': np.random.normal(7500, 2500, n_customers),
            'c_mktsegment': np.random.choice(['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD'], n_customers),
            'c_comment': [f"Comment for customer {i}" for i in range(1, n_customers + 1)]
        }
        
        customer_df = pd.DataFrame(customers)
        temp_path = os.path.join(self.temp_dir, "customer.parquet")
        customer_df.to_parquet(temp_path)
        self.upload_file_to_s3(temp_path, "tpch/parquet/sf10/customer.parquet")
        os.remove(temp_path)
        
        # Create nation table
        nations = {
            'n_nationkey': range(25),
            'n_name': ['ALGERIA', 'ARGENTINA', 'BRAZIL', 'CANADA', 'EGYPT', 'ETHIOPIA', 'FRANCE', 
                      'GERMANY', 'INDIA', 'INDONESIA', 'IRAN', 'IRAQ', 'JAPAN', 'JORDAN', 'KENYA',
                      'MOROCCO', 'MOZAMBIQUE', 'PERU', 'CHINA', 'ROMANIA', 'SAUDI ARABIA', 'VIETNAM',
                      'RUSSIA', 'UNITED KINGDOM', 'UNITED STATES'],
            'n_regionkey': [0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2, 4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1],
            'n_comment': [f"Comment for nation {i}" for i in range(25)]
        }
        
        nation_df = pd.DataFrame(nations)
        temp_path = os.path.join(self.temp_dir, "nation.parquet")
        nation_df.to_parquet(temp_path)
        self.upload_file_to_s3(temp_path, "tpch/parquet/sf10/nation.parquet")
        os.remove(temp_path)
        
        # Create orders table (larger dataset)
        n_orders = 3000000
        orders = {
            'o_orderkey': range(1, n_orders + 1),
            'o_custkey': np.random.randint(1, n_customers + 1, n_orders),
            'o_orderstatus': np.random.choice(['O', 'F', 'P'], n_orders, p=[0.5, 0.3, 0.2]),
            'o_totalprice': np.random.lognormal(8, 1, n_orders),
            'o_orderdate': [(datetime.now() - timedelta(days=np.random.randint(0, 2557))).strftime('%Y-%m-%d') 
                           for _ in range(n_orders)],
            'o_orderpriority': np.random.choice(['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW'], n_orders),
            'o_clerk': [f"Clerk#{i:09d}" for i in np.random.randint(1, 1000, n_orders)],
            'o_shippriority': np.random.randint(0, 2, n_orders),
            'o_comment': [f"Order comment {i}" for i in range(1, n_orders + 1)]
        }
        
        orders_df = pd.DataFrame(orders)
        temp_path = os.path.join(self.temp_dir, "orders.parquet")
        orders_df.to_parquet(temp_path)
        self.upload_file_to_s3(temp_path, "tpch/parquet/sf10/orders.parquet")
        os.remove(temp_path)
        
        # Create lineitem table (realistic scale)
        n_lineitems = 15000000
        lineitems = {
            'l_orderkey': np.random.randint(1, n_orders + 1, n_lineitems),
            'l_partkey': np.random.randint(1, 200000, n_lineitems),
            'l_suppkey': np.random.randint(1, 10000, n_lineitems),
            'l_linenumber': np.random.randint(1, 8, n_lineitems),
            'l_quantity': np.random.randint(1, 51, n_lineitems),
            'l_extendedprice': np.random.lognormal(6, 1, n_lineitems),
            'l_discount': np.random.uniform(0, 0.1, n_lineitems),
            'l_tax': np.random.uniform(0, 0.08, n_lineitems),
            'l_returnflag': np.random.choice(['A', 'N', 'R'], n_lineitems, p=[0.5, 0.25, 0.25]),
            'l_linestatus': np.random.choice(['O', 'F'], n_lineitems, p=[0.5, 0.5]),
            'l_shipdate': [(datetime.now() - timedelta(days=np.random.randint(0, 2557))).strftime('%Y-%m-%d') 
                          for _ in range(n_lineitems)],
            'l_commitdate': [(datetime.now() - timedelta(days=np.random.randint(0, 2557))).strftime('%Y-%m-%d') 
                           for _ in range(n_lineitems)],
            'l_receiptdate': [(datetime.now() - timedelta(days=np.random.randint(0, 2557))).strftime('%Y-%m-%d') 
                            for _ in range(n_lineitems)],
            'l_shipinstruct': np.random.choice(['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN'], n_lineitems),
            'l_shipmode': np.random.choice(['TRUCK', 'MAIL', 'REG AIR', 'SHIP', 'AIR', 'FOB', 'RAIL'], n_lineitems),
            'l_comment': [f"Lineitem comment {i}" for i in range(1, n_lineitems + 1)]
        }
        
        lineitem_df = pd.DataFrame(lineitems)
        temp_path = os.path.join(self.temp_dir, "lineitem.parquet")
        lineitem_df.to_parquet(temp_path)
        self.upload_file_to_s3(temp_path, "tpch/parquet/sf10/lineitem.parquet")
        os.remove(temp_path)
    
    def create_imagenette_placeholder(self):
        """Create placeholder structure for ImageNette data."""
        logger.info("Creating ImageNette placeholder...")
        
        # Note: We can't easily generate actual images, so we'll create a note
        # that the ImageNette dataset should be downloaded separately
        info_text = """
        The ImageNette dataset should be downloaded from:
        https://github.com/fastai/imagenette
        
        This is a subset of ImageNet with 10 classes.
        After downloading, extract to s3://ray-benchmark-data/imagenette2/train/
        
        For testing purposes, you can use any image dataset with the same structure.
        """
        
        temp_path = os.path.join(self.temp_dir, "README.txt")
        with open(temp_path, 'w') as f:
            f.write(info_text)
        
        self.upload_file_to_s3(temp_path, "imagenette2/README.txt")
        os.remove(temp_path)
    
    def create_text_captions(self):
        """Create text captions for multimodal AI."""
        logger.info("Creating text captions dataset...")
        
        captions = [
            "A beautiful sunset over the mountains",
            "People walking in a busy city street",
            "A cat sleeping on a window sill",
            "Fresh fruits arranged on a wooden table",
            "A vintage car parked on a cobblestone road",
            "Children playing in a park with trees",
            "Ocean waves crashing on rocky cliffs",
            "A modern skyscraper against blue sky",
            "Flowers blooming in a garden",
            "An old bridge crossing a river"
        ] * 500  # Repeat to get 5000 captions
        
        # Add some variation
        extended_captions = []
        for i, caption in enumerate(captions):
            if i % 100 == 0:
                extended_captions.append(f"{caption} on a sunny day")
            elif i % 100 == 1:
                extended_captions.append(f"{caption} in black and white")
            else:
                extended_captions.append(caption)
        
        temp_path = os.path.join(self.temp_dir, "captions.txt")
        with open(temp_path, 'w') as f:
            for caption in extended_captions:
                f.write(caption + '\n')
        
        self.upload_file_to_s3(temp_path, "text/captions.txt")
        os.remove(temp_path)
    
    def create_support_tickets(self):
        """Create support tickets for NLP analysis."""
        logger.info("Creating support tickets dataset...")
        
        ticket_templates = [
            {"category": "technical", "priority": "high", "text": "The application keeps crashing when I try to upload files"},
            {"category": "billing", "priority": "medium", "text": "I was charged twice for my subscription this month"},
            {"category": "feature", "priority": "low", "text": "Can you add dark mode to the application?"},
            {"category": "technical", "priority": "critical", "text": "Database connection is failing and users cannot log in"},
            {"category": "account", "priority": "medium", "text": "I forgot my password and cannot reset it"},
            {"category": "billing", "priority": "high", "text": "My payment method was declined but I have sufficient funds"},
            {"category": "technical", "priority": "low", "text": "The search function is running very slowly"},
            {"category": "feature", "priority": "medium", "text": "Please add two-factor authentication"},
            {"category": "account", "priority": "high", "text": "My account was suspended without explanation"},
            {"category": "technical", "priority": "medium", "text": "Email notifications are not being sent"}
        ]
        
        tickets = []
        for i in range(10000):
            template = random.choice(ticket_templates)
            ticket = {
                "ticket_id": f"TICKET-{i+1:06d}",
                "category": template["category"],
                "priority": template["priority"],
                "subject": template["text"],
                "description": f"{template['text']} - Additional details for ticket {i+1}",
                "created_at": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                "status": random.choice(["open", "in_progress", "resolved", "closed"])
            }
            tickets.append(ticket)
        
        # Create directory structure
        os.makedirs(os.path.join(self.temp_dir, "support"), exist_ok=True)
        temp_path = os.path.join(self.temp_dir, "support", "tickets.json")
        
        with open(temp_path, 'w') as f:
            for ticket in tickets:
                f.write(json.dumps(ticket) + '\n')
        
        self.upload_file_to_s3(temp_path, "support/tickets/tickets.json")
        os.remove(temp_path)
    
    def upload_all_datasets(self):
        """Upload all required datasets."""
        logger.info("Starting upload of all benchmark datasets...")
        
        # Upload existing parquet files (converted to required formats)
        self.upload_existing_parquet_files()
        
        # Create and upload generated datasets
        self.create_financial_data()
        self.create_titanic_dataset()
        self.create_nyc_taxi_data()
        self.create_tpch_data()
        self.create_imagenette_placeholder()
        self.create_text_captions()
        self.create_support_tickets()
        
        logger.info("All datasets uploaded successfully!")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info("You can now run the Ray Data templates with s3://ray-benchmark-data datasets")

def main():
    """Main function to run the upload process."""
    uploader = BenchmarkDataUploader()
    
    try:
        uploader.upload_all_datasets()
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
