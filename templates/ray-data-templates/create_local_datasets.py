#!/usr/bin/env python3
"""
Create Local Dataset Structure

This script creates all required datasets in a local directory structure
that matches the expected S3 paths. This allows for easy bulk upload later
when proper AWS credentials are available.

Usage:
    python create_local_datasets.py
    
Output:
    ./benchmark_data/ directory with all datasets organized by S3 path
"""

import os
import requests
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import shutil

# Local output directory
OUTPUT_DIR = "./benchmark_data"
BASE_PATH = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"

class LocalDatasetCreator:
    """Creates all required datasets in local directory structure."""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.base_path = Path(BASE_PATH)
        print(f"Creating datasets in: {self.output_dir.absolute()}")
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "financial").mkdir(exist_ok=True)
        (self.output_dir / "ml-datasets").mkdir(exist_ok=True)
        (self.output_dir / "nyc-taxi").mkdir(exist_ok=True)
        (self.output_dir / "tpch" / "parquet" / "sf10").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "medical" / "hl7-messages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "support" / "tickets").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "catalog").mkdir(exist_ok=True)
        (self.output_dir / "imagenette2").mkdir(exist_ok=True)
    
    def copy_existing_files(self):
        """Copy existing parquet files to appropriate locations."""
        print("Copying existing parquet files...")
        
        # Medical files
        medical_files = [
            ("ray-data-medical-connectors/laboratory_results.parquet", "medical/laboratory_results.parquet"),
            ("ray-data-medical-connectors/patient_medical_records.parquet", "medical/patient_medical_records.parquet"),
            ("ray-data-medical-connectors/dicom_imaging_metadata.parquet", "medical/dicom_imaging_metadata.parquet"),
            ("ray-data-medical-connectors/hl7_medical_messages.parquet", "medical/hl7_medical_messages.parquet"),
        ]
        
        # Log files
        log_files = [
            ("ray-data-log-ingestion/apache_access_logs.parquet", "logs/apache_access_logs.parquet"),
            ("ray-data-log-ingestion/application_logs.parquet", "logs/application_logs.parquet"),
            ("ray-data-log-ingestion/security_logs.parquet", "logs/security_logs.parquet"),
        ]
        
        # Quality files
        quality_files = [
            ("ray-data-data-quality-monitoring/ecommerce_customers_with_quality_issues.parquet", "catalog/customer_data.parquet"),
        ]
        
        all_files = medical_files + log_files + quality_files
        
        for local_rel_path, dest_path in all_files:
            src_path = self.base_path / local_rel_path
            dest_full_path = self.output_dir / dest_path
            
            if src_path.exists():
                shutil.copy2(src_path, dest_full_path)
                print(f"  ✅ Copied {src_path.name} -> {dest_path}")
            else:
                print(f"  ❌ Not found: {src_path}")
    
    def create_titanic_dataset(self):
        """Download or create Titanic dataset."""
        print("Creating Titanic dataset...")
        
        try:
            # Try to download real Titanic dataset
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            response = requests.get(url)
            
            if response.status_code == 200:
                output_path = self.output_dir / "ml-datasets" / "titanic.csv"
                with open(output_path, 'w') as f:
                    f.write(response.text)
                print(f"  ✅ Downloaded real Titanic dataset -> {output_path}")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ⚠️  Failed to download Titanic dataset: {e}")
            self._create_simple_titanic_csv()
    
    def _create_simple_titanic_csv(self):
        """Create a simple Titanic CSV."""
        print("  📝 Creating synthetic Titanic dataset...")
        
        csv_content = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S"""
        
        lines = csv_content.split('\n')
        header = lines[0]
        data_lines = lines[1:]
        
        output_path = self.output_dir / "ml-datasets" / "titanic.csv"
        with open(output_path, 'w') as f:
            f.write(header + '\n')
            for i in range(891):
                base_line = data_lines[i % len(data_lines)]
                modified_line = base_line.replace(base_line.split(',')[0], str(i+1))
                f.write(modified_line + '\n')
        
        print(f"  ✅ Created synthetic Titanic dataset -> {output_path}")
    
    def create_text_files(self):
        """Create text datasets."""
        print("Creating text datasets...")
        
        # Text captions
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
        ]
        
        output_path = self.output_dir / "text" / "captions.txt"
        with open(output_path, 'w') as f:
            for i in range(5000):
                caption = captions[i % len(captions)]
                f.write(f"{caption} variation {i}\n")
        print(f"  ✅ Created text captions -> {output_path}")
        
        # Apache access logs
        log_entries = [
            '127.0.0.1 - - [10/Oct/2023:13:55:36 +0000] "GET /index.html HTTP/1.1" 200 2326',
            '127.0.0.1 - - [10/Oct/2023:13:55:37 +0000] "POST /login HTTP/1.1" 302 0',
            '192.168.1.100 - - [10/Oct/2023:13:55:38 +0000] "GET /dashboard HTTP/1.1" 200 5432',
            '10.0.0.1 - - [10/Oct/2023:13:55:39 +0000] "GET /api/users HTTP/1.1" 200 1024',
            '203.0.113.1 - - [10/Oct/2023:13:55:40 +0000] "POST /api/data HTTP/1.1" 201 256',
        ]
        
        output_path = self.output_dir / "logs" / "apache-access.log"
        with open(output_path, 'w') as f:
            for i in range(10000):
                entry = log_entries[i % len(log_entries)]
                f.write(entry.replace("13:55:36", f"13:{55 + (i % 60):02d}:{36 + (i % 24):02d}") + '\n')
        print(f"  ✅ Created Apache logs -> {output_path}")
        
        # Security logs
        security_entries = [
            "2023-10-10T13:55:36Z SECURITY: Failed login attempt from 192.168.1.100",
            "2023-10-10T13:55:37Z SECURITY: Successful authentication for user admin",
            "2023-10-10T13:55:38Z SECURITY: Firewall blocked connection from 203.0.113.1",
            "2023-10-10T13:55:39Z SECURITY: Privilege escalation detected for user guest",
            "2023-10-10T13:55:40Z SECURITY: New device detected for user admin",
        ]
        
        output_path = self.output_dir / "logs" / "security.log"
        with open(output_path, 'w') as f:
            for i in range(5000):
                entry = security_entries[i % len(security_entries)]
                f.write(entry + '\n')
        print(f"  ✅ Created security logs -> {output_path}")
    
    def create_json_files(self):
        """Create JSON datasets."""
        print("Creating JSON datasets...")
        
        # Application logs
        output_path = self.output_dir / "logs" / "application.json"
        with open(output_path, 'w') as f:
            for i in range(5000):
                log_entry = {
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "level": random.choice(["INFO", "WARN", "ERROR", "DEBUG"]),
                    "service": random.choice(["auth", "api", "db", "cache"]),
                    "message": f"Application event {i}",
                    "request_id": f"req-{i:06d}",
                    "user_id": random.randint(1000, 9999) if i % 3 == 0 else None
                }
                f.write(json.dumps(log_entry) + '\n')
        print(f"  ✅ Created application logs -> {output_path}")
        
        # DICOM metadata
        output_path = self.output_dir / "medical" / "dicom-metadata.json"
        with open(output_path, 'w') as f:
            for i in range(1000):
                dicom_record = {
                    "study_id": f"STUDY-{i:06d}",
                    "patient_id": f"PAT-{i:04d}",
                    "modality": random.choice(["CT", "MRI", "XRAY", "ULTRASOUND"]),
                    "study_date": (datetime.now() - timedelta(days=i)).strftime("%Y%m%d"),
                    "body_part": random.choice(["CHEST", "HEAD", "ABDOMEN", "PELVIS"]),
                    "slice_count": random.randint(10, 500),
                    "file_size_mb": random.randint(10, 1000)
                }
                f.write(json.dumps(dicom_record) + '\n')
        print(f"  ✅ Created DICOM metadata -> {output_path}")
        
        # Support tickets
        output_path = self.output_dir / "support" / "tickets" / "tickets.json"
        with open(output_path, 'w') as f:
            categories = ["technical", "billing", "feature", "account"]
            priorities = ["low", "medium", "high", "critical"]
            
            for i in range(10000):
                ticket = {
                    "ticket_id": f"TICKET-{i+1:06d}",
                    "category": random.choice(categories),
                    "priority": random.choice(priorities),
                    "subject": f"Support request {i+1}",
                    "description": f"Detailed description of issue {i+1}",
                    "created_at": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                    "status": random.choice(["open", "in_progress", "resolved", "closed"])
                }
                f.write(json.dumps(ticket) + '\n')
        print(f"  ✅ Created support tickets -> {output_path}")
    
    def create_csv_files(self):
        """Create CSV datasets."""
        print("Creating CSV datasets...")
        
        # Patient records
        output_path = self.output_dir / "medical" / "patient-records.csv"
        with open(output_path, 'w') as f:
            f.write("patient_id,age,gender,diagnosis,admission_date,discharge_date,department\n")
            
            for i in range(5000):
                patient = {
                    "patient_id": f"PAT-{i+1:06d}",
                    "age": random.randint(18, 90),
                    "gender": random.choice(["M", "F"]),
                    "diagnosis": random.choice(["Hypertension", "Diabetes", "Pneumonia", "Fracture", "Cardiac"]),
                    "admission_date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                    "discharge_date": (datetime.now() - timedelta(days=random.randint(0, 360))).strftime("%Y-%m-%d"),
                    "department": random.choice(["Cardiology", "Emergency", "Surgery", "Internal", "Pediatrics"])
                }
                
                line = f"{patient['patient_id']},{patient['age']},{patient['gender']},{patient['diagnosis']},{patient['admission_date']},{patient['discharge_date']},{patient['department']}\n"
                f.write(line)
        print(f"  ✅ Created patient records -> {output_path}")
    
    def create_hl7_files(self):
        """Create HL7 message files."""
        print("Creating HL7 datasets...")
        
        output_path = self.output_dir / "medical" / "hl7-messages" / "messages.hl7"
        with open(output_path, 'w') as f:
            for i in range(1000):
                hl7_message = f"""MSH|^~\\&|SYSTEM|HOSPITAL|RECEIVER|CLINIC|{datetime.now().strftime('%Y%m%d%H%M%S')}||ADT^A01|{i+1:06d}|P|2.4
EVN||{datetime.now().strftime('%Y%m%d%H%M%S')}
PID|||PAT-{i+1:06d}||PATIENT^TEST||{(datetime.now() - timedelta(days=random.randint(365*20, 365*80))).strftime('%Y%m%d')}|M
PV1||I|ICU^101^01||||||||||||||V
"""
                f.write(hl7_message + '\n')
        print(f"  ✅ Created HL7 messages -> {output_path}")
    
    def create_info_files(self):
        """Create informational files."""
        print("Creating info files...")
        
        # ImageNette placeholder
        info_text = """
The ImageNette dataset should be downloaded from:
https://github.com/fastai/imagenette

This is a subset of ImageNet with 10 classes.
After downloading, upload to s3://ray-benchmark-data/imagenette2/train/

To download and upload ImageNette:
1. wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
2. tar -xzf imagenette2.tgz  
3. aws s3 sync imagenette2/train/ s3://ray-benchmark-data/imagenette2/train/
"""
        
        output_path = self.output_dir / "imagenette2" / "README.txt"
        with open(output_path, 'w') as f:
            f.write(info_text)
        print(f"  ✅ Created ImageNette info -> {output_path}")
    
    def create_all_datasets(self):
        """Create all datasets locally."""
        print("=" * 60)
        print("CREATING LOCAL BENCHMARK DATASETS")
        print("=" * 60)
        
        self.copy_existing_files()
        self.create_titanic_dataset()
        self.create_text_files()
        self.create_json_files()
        self.create_csv_files()
        self.create_hl7_files()
        self.create_info_files()
        
        print("\n" + "=" * 60)
        print("DATASET CREATION COMPLETE")
        print("=" * 60)
        
        # Show summary
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
        file_count = len(list(self.output_dir.rglob('*')))
        
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"📊 Total files created: {file_count}")
        print(f"💾 Total size: {total_size / (1024*1024):.1f} MB")
        
        print("\n📂 Directory structure:")
        for item in sorted(self.output_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(self.output_dir)
                size_mb = item.stat().st_size / (1024*1024)
                print(f"  {rel_path} ({size_mb:.1f} MB)")
        
        print(f"\n🚀 To upload to S3:")
        print(f"  aws s3 sync {self.output_dir} s3://ray-benchmark-data/")
        print(f"\n📋 Individual file uploads:")
        for item in sorted(self.output_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(self.output_dir)
                print(f"  aws s3 cp {item} s3://ray-benchmark-data/{rel_path}")

def main():
    """Main function."""
    creator = LocalDatasetCreator()
    creator.create_all_datasets()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
