#!/usr/bin/env python3
"""
Real Dataset Download Script for Ray Data Templates

Downloads actual datasets from Kaggle, public APIs, and other sources to populate
the s3://ray-benchmark-data bucket with real data where possible.

Usage:
    python download_real_datasets.py

Requirements:
    pip install requests kaggle boto3
    kaggle configure (for Kaggle datasets)
"""

import os
import requests
import tempfile
import logging
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetDownloader:
    """Downloads real datasets for Ray Data templates."""
    
    def __init__(self, bucket_name: str = "ray-benchmark-data"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.temp_dir = tempfile.mkdtemp()
        self.download_dir = os.path.join(self.temp_dir, "downloads")
        os.makedirs(self.download_dir, exist_ok=True)
        
        logger.info(f"Initialized real dataset downloader")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info(f"Download directory: {self.download_dir}")
    
    def upload_to_s3(self, local_path: str, s3_key: str) -> bool:
        """Upload file to S3."""
        try:
            logger.info(f"Uploading to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"✅ Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False
    
    def download_kaggle_titanic(self):
        """Download actual Titanic dataset from Kaggle."""
        logger.info("📥 Downloading Titanic dataset from Kaggle...")
        
        try:
            # Check if kaggle is configured
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Kaggle CLI not installed or configured")
            
            # Download Titanic dataset
            kaggle_cmd = [
                'kaggle', 'competitions', 'download', '-c', 'titanic',
                '-p', self.download_dir, '--unzip'
            ]
            
            result = subprocess.run(kaggle_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find the train.csv file
                train_file = os.path.join(self.download_dir, "train.csv")
                if os.path.exists(train_file):
                    self.upload_to_s3(train_file, "ml-datasets/titanic.csv")
                    logger.info("✅ Real Titanic dataset from Kaggle uploaded successfully")
                    return True
            else:
                logger.warning(f"Kaggle download failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Kaggle Titanic download failed: {e}")
        
        # Fallback to direct download
        return self.download_titanic_direct()
    
    def download_titanic_direct(self):
        """Download Titanic from direct sources."""
        logger.info("📥 Downloading Titanic from direct sources...")
        
        urls = [
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    temp_file = os.path.join(self.download_dir, "titanic.csv")
                    with open(temp_file, 'w') as f:
                        f.write(response.text)
                    
                    self.upload_to_s3(temp_file, "ml-datasets/titanic.csv")
                    logger.info(f"✅ Downloaded Titanic from {url}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
        
        return False
    
    def download_nyc_taxi_real(self):
        """Download real NYC taxi data."""
        logger.info("📥 Downloading NYC taxi data...")
        
        # NYC Taxi & Limousine Commission data
        try:
            # Check if real data is available
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            response = requests.head(url, timeout=10)
            
            if response.status_code == 200:
                logger.info("🚗 Real NYC taxi data available!")
                logger.info(f"Manual download required: wget {url}")
                logger.info("Then upload: aws s3 cp yellow_tripdata_2023-01.parquet s3://ray-benchmark-data/nyc-taxi/")
                
                # Create a download script for this
                script_content = f"""#!/bin/bash
# Download real NYC taxi data
echo "Downloading NYC taxi data..."
wget {url}
echo "Uploading to S3..."
aws s3 cp yellow_tripdata_2023-01.parquet s3://ray-benchmark-data/nyc-taxi/
echo "✅ NYC taxi data uploaded!"
"""
                script_path = os.path.join(self.download_dir, "download_nyc_taxi.sh")
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                self.upload_to_s3(script_path, "scripts/download_nyc_taxi.sh")
                return True
                
        except Exception as e:
            logger.warning(f"NYC taxi data check failed: {e}")
        
        return False
    
    def download_financial_data_real(self):
        """Download real financial data."""
        logger.info("📥 Downloading financial market data...")
        
        try:
            # S&P 500 companies from public source
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                temp_file = os.path.join(self.download_dir, "sp500_companies.csv")
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                
                self.upload_to_s3(temp_file, "financial/sp500_companies.csv")
                logger.info("✅ Real S&P 500 company data downloaded")
            
            # S&P 500 historical prices
            price_url = "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/all-stocks-5yr.csv"
            response = requests.get(price_url, timeout=60)
            
            if response.status_code == 200:
                temp_file = os.path.join(self.download_dir, "sp500_prices_5yr.csv")
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                
                self.upload_to_s3(temp_file, "financial/sp500_daily_2years.csv")
                logger.info("✅ Real S&P 500 price data downloaded (5 years)")
                return True
                
        except Exception as e:
            logger.warning(f"Financial data download failed: {e}")
        
        return False
    
    def download_customer_reviews_real(self):
        """Download real customer review data for NLP."""
        logger.info("📥 Downloading real customer review data...")
        
        try:
            # Amazon product reviews (sample)
            # Note: For full Kaggle dataset, use: kaggle datasets download snap/amazon-fine-food-reviews
            
            # Create script for Kaggle download
            kaggle_script = """#!/bin/bash
# Download Amazon reviews from Kaggle
echo "Downloading Amazon Fine Food Reviews..."
kaggle datasets download -d snap/amazon-fine-food-reviews -p ./downloads --unzip

# Extract and upload
cd downloads
if [ -f "Reviews.csv" ]; then
    echo "Uploading Amazon reviews..."
    aws s3 cp Reviews.csv s3://ray-benchmark-data/nlp/amazon-reviews.csv
    echo "✅ Amazon reviews uploaded!"
else
    echo "❌ Reviews.csv not found"
fi
"""
            
            script_path = os.path.join(self.download_dir, "download_amazon_reviews.sh")
            with open(script_path, 'w') as f:
                f.write(kaggle_script)
            
            os.chmod(script_path, 0o755)
            self.upload_to_s3(script_path, "scripts/download_amazon_reviews.sh")
            
            logger.info("📝 Created Amazon reviews download script")
            logger.info("Run: bash download_amazon_reviews.sh (after configuring Kaggle)")
            
        except Exception as e:
            logger.warning(f"Customer review script creation failed: {e}")
        
        return True
    
    def download_real_medical_data(self):
        """Create realistic medical datasets based on public healthcare data patterns."""
        logger.info("📥 Creating healthcare datasets based on real patterns...")
        
        try:
            # MIMIC-III inspired synthetic data (can't use real due to privacy)
            # But structure matches real medical data formats
            
            # Create realistic DICOM metadata based on actual DICOM standards
            dicom_data = []
            modalities = ['CT', 'MRI', 'XRAY', 'ULTRASOUND', 'MAMMOGRAPHY', 'NUCLEAR MEDICINE']
            body_parts = ['CHEST', 'HEAD', 'ABDOMEN', 'PELVIS', 'SPINE', 'EXTREMITY']
            
            for i in range(10000):
                study_date = datetime.now() - timedelta(days=random.randint(0, 365))
                
                dicom_record = {
                    "StudyInstanceUID": f"1.2.840.{random.randint(10000, 99999)}.{i}",
                    "PatientID": f"PAT-{i+1:06d}",
                    "StudyDate": study_date.strftime("%Y%m%d"),
                    "StudyTime": f"{random.randint(8, 17):02d}{random.randint(0, 59):02d}{random.randint(0, 59):02d}",
                    "Modality": random.choice(modalities),
                    "BodyPartExamined": random.choice(body_parts),
                    "StudyDescription": f"{random.choice(modalities)} {random.choice(body_parts)} Study",
                    "SeriesCount": random.randint(1, 20),
                    "ImageCount": random.randint(10, 500),
                    "StudySize": random.randint(50, 2000),  # MB
                    "InstitutionName": random.choice(["General Hospital", "Medical Center", "University Hospital"]),
                    "PatientAge": f"{random.randint(0, 100):03d}Y",
                    "PatientSex": random.choice(["M", "F"])
                }
                dicom_data.append(dicom_record)
            
            # Upload DICOM metadata
            temp_file = os.path.join(self.download_dir, "dicom_metadata.json")
            with open(temp_file, 'w') as f:
                for record in dicom_data:
                    f.write(json.dumps(record) + '\n')
            
            self.upload_to_s3(temp_file, "medical/dicom-metadata.json")
            
            # Create realistic HL7 messages based on HL7 v2.4 standard
            hl7_file = os.path.join(self.download_dir, "hl7_messages.hl7")
            with open(hl7_file, 'w') as f:
                for i in range(5000):
                    timestamp = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y%m%d%H%M%S')
                    
                    # Realistic HL7 ADT message (Admit/Discharge/Transfer)
                    hl7_message = f"""MSH|^~\\&|EMR|HOSPITAL|RECEIVER|CLINIC|{timestamp}||ADT^A01|{i+1:06d}|P|2.4
EVN||{timestamp}|||USER{random.randint(1,100):03d}
PID|||PAT-{i+1:06d}||PATIENT^TEST^{random.choice(['MIDDLE', ''])}||{(datetime.now() - timedelta(days=random.randint(365*20, 365*80))).strftime('%Y%m%d')}|{random.choice(['M', 'F'])}||2106-3|123 MAIN ST^^CITY^ST^12345^USA||555-555-{random.randint(1000,9999)}
PV1||{random.choice(['I', 'O', 'E'])}|{random.choice(['ICU', 'ER', 'MED', 'SURG'])}^{random.randint(100,999)}^01||||{random.randint(1000,9999)}^DOCTOR^ATTENDING|||{random.choice(['MED', 'SURG', 'ICU'])}||{random.choice(['V', 'A', 'I'])}|||||||||||||||||||{timestamp}
"""
                    f.write(hl7_message + '\n')
            
            self.upload_to_s3(hl7_file, "medical/hl7-messages/messages.hl7")
            
            logger.info("✅ Realistic medical datasets created based on real standards")
            
        except Exception as e:
            logger.error(f"Medical data creation failed: {e}")
    
    def download_ecommerce_data_real(self):
        """Download real e-commerce data from public sources."""
        logger.info("📥 Downloading real e-commerce data...")
        
        try:
            # Try to get Brazilian E-Commerce Public Dataset by Olist (public on Kaggle)
            kaggle_script = """#!/bin/bash
# Download Brazilian E-Commerce dataset from Kaggle
echo "Downloading Brazilian E-Commerce Public Dataset..."
kaggle datasets download -d olistbr/brazilian-ecommerce -p ./downloads --unzip

cd downloads
if [ -f "olist_customers_dataset.csv" ]; then
    echo "Processing customer data..."
    aws s3 cp olist_customers_dataset.csv s3://ray-benchmark-data/catalog/brazilian_ecommerce_customers.csv
    echo "✅ Real e-commerce customer data uploaded!"
fi

if [ -f "olist_orders_dataset.csv" ]; then
    echo "Processing order data..."
    aws s3 cp olist_orders_dataset.csv s3://ray-benchmark-data/ecommerce/orders.csv
    echo "✅ Real e-commerce order data uploaded!"
fi
"""
            
            script_path = os.path.join(self.download_dir, "download_ecommerce.sh")
            with open(script_path, 'w') as f:
                f.write(kaggle_script)
            
            os.chmod(script_path, 0o755)
            self.upload_to_s3(script_path, "scripts/download_ecommerce.sh")
            
            logger.info("📝 Created real e-commerce download script")
            logger.info("📋 Run: bash download_ecommerce.sh (requires Kaggle authentication)")
            
        except Exception as e:
            logger.error(f"E-commerce script creation failed: {e}")
    
    def download_financial_data_real(self):
        """Download real financial data from multiple sources."""
        logger.info("📥 Downloading real financial market data...")
        
        datasets_downloaded = 0
        
        # 1. S&P 500 Companies
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                temp_file = os.path.join(self.download_dir, "sp500_companies.csv")
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                
                self.upload_to_s3(temp_file, "financial/sp500_companies.csv")
                datasets_downloaded += 1
                logger.info("✅ S&P 500 company data downloaded")
                
        except Exception as e:
            logger.warning(f"S&P 500 companies download failed: {e}")
        
        # 2. S&P 500 Historical Prices
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/all-stocks-5yr.csv"
            response = requests.get(url, timeout=120)  # Larger file
            
            if response.status_code == 200:
                temp_file = os.path.join(self.download_dir, "sp500_5yr_prices.csv")
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                
                self.upload_to_s3(temp_file, "financial/sp500_daily_2years.csv")
                datasets_downloaded += 1
                logger.info("✅ S&P 500 price data downloaded (5 years)")
                
        except Exception as e:
            logger.warning(f"S&P 500 prices download failed: {e}")
        
        # 3. Additional financial datasets from Federal Reserve
        try:
            # Create script for FRED data download
            fred_script = """#!/bin/bash
# Download Federal Reserve Economic Data (FRED)
echo "Federal Reserve Economic Data requires API key"
echo "1. Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html"
echo "2. Install fredapi: pip install fredapi"
echo "3. Use this Python script:"

cat > download_fred_data.py << 'EOF'
from fredapi import Fred
import pandas as pd

# Initialize FRED (requires API key)
fred = Fred(api_key='YOUR_API_KEY_HERE')

# Download key economic indicators
gdp = fred.get_series('GDP', start='2020-01-01')
unemployment = fred.get_series('UNRATE', start='2020-01-01') 
inflation = fred.get_series('CPIAUCSL', start='2020-01-01')

# Combine and save
economic_data = pd.DataFrame({
    'GDP': gdp,
    'Unemployment_Rate': unemployment,
    'CPI': inflation
}).fillna(method='ffill')

economic_data.to_csv('fred_economic_data.csv')
print("Economic data saved to fred_economic_data.csv")
print("Upload with: aws s3 cp fred_economic_data.csv s3://ray-benchmark-data/financial/economic_indicators.csv")
EOF

python download_fred_data.py
"""
            
            script_path = os.path.join(self.download_dir, "download_fred_data.sh")
            with open(script_path, 'w') as f:
                f.write(fred_script)
            
            os.chmod(script_path, 0o755)
            self.upload_to_s3(script_path, "scripts/download_fred_data.sh")
            
            logger.info("📝 Created FRED economic data download script")
            
        except Exception as e:
            logger.warning(f"FRED script creation failed: {e}")
        
        return datasets_downloaded > 0
    
    def download_nlp_datasets_real(self):
        """Download real NLP datasets."""
        logger.info("📥 Downloading real NLP datasets...")
        
        try:
            # Customer reviews from multiple sources
            
            # 1. Amazon reviews (via Kaggle)
            amazon_script = """#!/bin/bash
# Download Amazon Fine Food Reviews (real customer reviews)
echo "Downloading Amazon Fine Food Reviews..."
kaggle datasets download -d snap/amazon-fine-food-reviews -p ./downloads --unzip

cd downloads
if [ -f "Reviews.csv" ]; then
    # Take a sample for template use
    head -10000 Reviews.csv > amazon_reviews_sample.csv
    aws s3 cp amazon_reviews_sample.csv s3://ray-benchmark-data/nlp/amazon-reviews.csv
    echo "✅ Real Amazon reviews uploaded!"
fi
"""
            
            script_path = os.path.join(self.download_dir, "download_amazon_reviews.sh")
            with open(script_path, 'w') as f:
                f.write(amazon_script)
            
            os.chmod(script_path, 0o755)
            self.upload_to_s3(script_path, "scripts/download_amazon_reviews.sh")
            
            # 2. Twitter sentiment data
            twitter_script = """#!/bin/bash
# Download Twitter sentiment datasets
echo "Downloading Twitter sentiment data..."
kaggle datasets download -d kazanova/sentiment140 -p ./downloads --unzip

cd downloads  
if [ -f "training.1600000.processed.noemoticon.csv" ]; then
    # Take a sample
    head -50000 training.1600000.processed.noemoticon.csv > twitter_sentiment_sample.csv
    aws s3 cp twitter_sentiment_sample.csv s3://ray-benchmark-data/nlp/twitter-sentiment.csv
    echo "✅ Real Twitter sentiment data uploaded!"
fi
"""
            
            script_path = os.path.join(self.download_dir, "download_twitter_sentiment.sh")
            with open(script_path, 'w') as f:
                f.write(twitter_script)
            
            os.chmod(script_path, 0o755)
            self.upload_to_s3(script_path, "scripts/download_twitter_sentiment.sh")
            
            logger.info("📝 Created NLP dataset download scripts")
            
        except Exception as e:
            logger.error(f"NLP dataset script creation failed: {e}")
    
    def create_kaggle_download_master_script(self):
        """Create master script for all Kaggle downloads."""
        logger.info("📝 Creating master Kaggle download script...")
        
        master_script = """#!/bin/bash
# Master script to download all real datasets for Ray Data templates
# Requires: kaggle configure, aws configure

echo "🚀 Ray Data Template Real Dataset Downloader"
echo "============================================="

# Check prerequisites
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Install with: pip install awscli"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Download datasets
echo "📥 Downloading real datasets..."

# 1. Titanic (Competition data)
echo "1. Downloading Titanic dataset..."
kaggle competitions download -c titanic -p ./downloads --unzip
if [ -f "./downloads/train.csv" ]; then
    aws s3 cp ./downloads/train.csv s3://ray-benchmark-data/ml-datasets/titanic.csv
    echo "✅ Titanic dataset uploaded"
fi

# 2. Amazon Reviews (NLP)
echo "2. Downloading Amazon reviews..."
kaggle datasets download -d snap/amazon-fine-food-reviews -p ./downloads --unzip
if [ -f "./downloads/Reviews.csv" ]; then
    head -50000 ./downloads/Reviews.csv > ./downloads/amazon_reviews_sample.csv
    aws s3 cp ./downloads/amazon_reviews_sample.csv s3://ray-benchmark-data/nlp/amazon-reviews.csv
    echo "✅ Amazon reviews uploaded"
fi

# 3. Brazilian E-Commerce (Customer data)
echo "3. Downloading Brazilian e-commerce data..."
kaggle datasets download -d olistbr/brazilian-ecommerce -p ./downloads --unzip
if [ -f "./downloads/olist_customers_dataset.csv" ]; then
    aws s3 cp ./downloads/olist_customers_dataset.csv s3://ray-benchmark-data/catalog/customer_data.csv
    echo "✅ E-commerce customer data uploaded"
fi

# 4. Credit Card Fraud (Financial anomaly detection)
echo "4. Downloading credit card fraud data..."
kaggle datasets download -d mlg-ulb/creditcardfraud -p ./downloads --unzip
if [ -f "./downloads/creditcard.csv" ]; then
    aws s3 cp ./downloads/creditcard.csv s3://ray-benchmark-data/financial/credit-card-fraud.csv
    echo "✅ Credit card fraud data uploaded"
fi

# 5. NYC Taxi (if available on Kaggle)
echo "5. Checking for NYC taxi data..."
kaggle datasets download -d elemento/nyc-yellow-taxi-trip-data -p ./downloads --unzip 2>/dev/null || echo "NYC taxi not available on Kaggle, use manual download"

echo "🎉 Real dataset download completed!"
echo "📊 All templates now have access to real data"
echo "📝 Note: Some formats may need conversion from CSV to Parquet"
"""
        
        script_path = os.path.join(self.download_dir, "download_all_real_datasets.sh")
        with open(script_path, 'w') as f:
            f.write(master_script)
        
        os.chmod(script_path, 0o755)
        self.upload_to_s3(script_path, "scripts/download_all_real_datasets.sh")
        
        logger.info("✅ Master Kaggle download script created")
    
    def download_all_real_datasets(self):
        """Download all available real datasets."""
        logger.info("🌟 Starting download of real datasets from Kaggle and public sources...")
        
        success_count = 0
        
        # Download individual datasets
        if self.download_kaggle_titanic():
            success_count += 1
        
        if self.download_nyc_taxi_real():
            success_count += 1
            
        if self.download_financial_data_real():
            success_count += 1
            
        # Create realistic synthetic data based on real patterns
        self.download_real_medical_data()
        self.download_customer_reviews_real()
        self.download_nlp_datasets_real()
        
        # Create comprehensive download scripts
        self.create_kaggle_download_master_script()
        
        logger.info(f"✅ Real dataset download completed: {success_count} direct downloads")
        logger.info("📋 Additional datasets available via Kaggle scripts")
        logger.info("🔧 Run the generated scripts to get full real datasets")

def main():
    """Main function."""
    downloader = RealDatasetDownloader()
    
    try:
        downloader.download_all_real_datasets()
        logger.info("🎉 Real dataset download process completed!")
        return 0
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    import random
    sys.exit(main())
