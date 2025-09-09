# Cross-Template Workflow Examples

This guide demonstrates how to combine multiple Ray Data templates to create comprehensive end-to-end data processing workflows. Each workflow shows practical business scenarios using multiple templates together.

## Workflow 1: Complete ML Pipeline

**Use Case**: Build an end-to-end machine learning pipeline from raw data to model deployment

**Templates Used**: Data Quality Monitoring → ML Feature Engineering → Batch Inference Optimization

**Business Scenario**: E-commerce company wants to predict customer churn using transaction data

### **Step 1: Data Quality Assessment**

```python
# Start with data quality monitoring
import ray
from ray.data import read_parquet

# Load customer transaction data
raw_data = read_parquet("s3://your-bucket/customer-transactions.parquet")

# Apply data quality monitoring (from data-quality-monitoring template)
class BusinessRuleEngine:
    def __call__(self, batch):
        validated_records = []
        for record in batch:
            # Apply business rules
            quality_score = 1.0
            if record.get('transaction_amount', 0) < 0:
                quality_score -= 0.3
            if not record.get('customer_id'):
                quality_score -= 0.5
                
            validated_records.append({
                **record,
                'quality_score': quality_score,
                'is_high_quality': quality_score >= 0.7
            })
        return validated_records

# Filter for high-quality data
validated_data = raw_data.map_batches(BusinessRuleEngine(), batch_size=1000)
clean_data = validated_data.filter(lambda x: x['is_high_quality'])

print(f"Data quality results: {clean_data.count()} high-quality records")
```

### **Step 2: Feature Engineering**

```python
# Apply ML feature engineering (from ml-feature-engineering template)
class CustomerFeatureEngineer:
    def __call__(self, batch):
        engineered_features = []
        for record in batch:
            # Create ML features
            transaction_amount = record.get('transaction_amount', 0)
            days_since_last = record.get('days_since_last_transaction', 0)
            
            engineered_record = {
                **record,
                'transaction_amount_log': np.log(transaction_amount + 1),
                'is_high_value': transaction_amount > 1000,
                'recency_score': 1.0 / (days_since_last + 1),
                'customer_segment': 'premium' if transaction_amount > 500 else 'standard'
            }
            engineered_features.append(engineered_record)
        return engineered_features

# Engineer features for ML
featured_data = clean_data.map_batches(CustomerFeatureEngineer(), batch_size=500)

print(f"Feature engineering complete: {featured_data.count()} records with ML features")
```

### **Step 3: Batch Inference**

```python
# Apply batch inference optimization (from batch-inference-optimization template)
import torch
from sklearn.ensemble import RandomForestClassifier

class ChurnPredictor:
    def __init__(self):
        # Load pre-trained churn model (simplified for demo)
        self.model = RandomForestClassifier(n_estimators=100)
        # In practice, load your trained model here
        
    def __call__(self, batch):
        predictions = []
        for record in batch:
            # Extract features for prediction
            features = [
                record.get('transaction_amount_log', 0),
                record.get('recency_score', 0),
                record.get('is_high_value', 0)
            ]
            
            # Simulate model prediction
            churn_probability = np.random.random()  # Replace with actual model.predict_proba()
            
            predictions.append({
                **record,
                'churn_probability': churn_probability,
                'churn_prediction': churn_probability > 0.5,
                'confidence': abs(churn_probability - 0.5) * 2
            })
        return predictions

# Apply churn prediction
churn_predictions = featured_data.map_batches(
    ChurnPredictor(),
    batch_size=100,
    concurrency=4
)

print(f"Churn predictions complete: {churn_predictions.count()} predictions")

# Display sample results
sample_predictions = churn_predictions.take(5)
for pred in sample_predictions:
    print(f"Customer {pred.get('customer_id')}: {pred['churn_probability']:.2f} churn probability")
```

## Workflow 2: Content Intelligence Pipeline

**Use Case**: Analyze multimodal social media content for brand monitoring

**Templates Used**: Multimodal AI Pipeline → NLP Text Analytics → Data Catalog

**Business Scenario**: Marketing team wants to analyze social media posts containing images, text, and sentiment

### **Step 1: Multimodal Content Processing**

```python
# Process multimodal content (from multimodal-ai-pipeline template)
from ray.data import read_images, read_text

# Load social media content
image_posts = read_images("s3://social-media-bucket/images/", mode="RGB")
text_posts = read_text("s3://social-media-bucket/captions/")

# Process images for visual content analysis
class ContentVisionProcessor:
    def __init__(self):
        import torch
        from torchvision import models
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
    
    def __call__(self, batch):
        visual_features = []
        for img in batch["image"]:
            # Extract visual features for content analysis
            # Simplified feature extraction
            visual_features.append({
                'has_people': np.random.random() > 0.5,  # Replace with actual detection
                'has_products': np.random.random() > 0.7,
                'image_quality_score': np.random.uniform(0.5, 1.0),
                'visual_complexity': np.random.uniform(0.1, 1.0)
            })
        return visual_features

processed_images = image_posts.map_batches(ContentVisionProcessor(), batch_size=8)
```

### **Step 2: Text Sentiment Analysis**

```python
# Apply NLP analysis (from nlp-text-analytics template)
from transformers import pipeline

class SocialMediaNLPProcessor:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
    
    def __call__(self, batch):
        nlp_results = []
        for item in batch:
            text = item.get('text', '')
            
            # Analyze sentiment and extract insights
            sentiment = self.sentiment_pipeline(text[:512])[0]
            
            # Extract brand mentions (simplified)
            brand_mentions = len([word for word in text.lower().split() 
                                if word in ['brand', 'product', 'company']])
            
            nlp_results.append({
                **item,
                'sentiment': sentiment['label'],
                'sentiment_confidence': sentiment['score'],
                'brand_mentions': brand_mentions,
                'is_brand_relevant': brand_mentions > 0,
                'text_length': len(text)
            })
        return nlp_results

# Process text content
processed_text = text_posts.map_batches(SocialMediaNLPProcessor(), batch_size=16)
```

### **Step 3: Catalog and Governance**

```python
# Catalog results (from enterprise-data-catalog template)
class ContentCatalogManager:
    def __init__(self):
        self.catalog = {}
    
    def __call__(self, batch):
        cataloged_content = []
        for item in batch:
            # Add metadata for content governance
            catalog_entry = {
                **item,
                'content_type': 'social_media_post',
                'processing_timestamp': datetime.now().isoformat(),
                'data_lineage': 'social_media → multimodal_ai → nlp_analysis',
                'governance_tags': ['public_content', 'brand_monitoring'],
                'retention_policy': '1_year',
                'access_level': 'marketing_team'
            }
            cataloged_content.append(catalog_entry)
        return cataloged_content

# Apply cataloging
final_content = processed_text.map_batches(ContentCatalogManager(), batch_size=50)

# Generate brand monitoring report
brand_relevant_content = final_content.filter(lambda x: x['is_brand_relevant'])
positive_sentiment = brand_relevant_content.filter(lambda x: x['sentiment'] == 'POSITIVE')

print(f"Brand monitoring results:")
print(f"  Total content analyzed: {final_content.count()}")
print(f"  Brand-relevant posts: {brand_relevant_content.count()}")
print(f"  Positive brand sentiment: {positive_sentiment.count()}")
```

## Workflow 3: Geospatial Business Intelligence

**Use Case**: Urban planning analysis combining multiple data sources

**Templates Used**: Geospatial Analysis → Log Ingestion → Time Series Forecasting

**Business Scenario**: City planning department analyzing traffic patterns, events, and trends

### **Step 1: Geospatial Data Processing**

```python
# Process spatial data (from geospatial-analysis template)
from ray.data import read_parquet

# Load urban planning datasets
traffic_data = read_parquet("s3://city-data/traffic-sensors.parquet")
event_logs = read_parquet("s3://city-data/event-logs.parquet")

# Analyze traffic patterns spatially
class TrafficAnalyzer:
    def __call__(self, batch):
        traffic_analysis = []
        for record in batch:
            lat, lon = record.get('latitude', 0), record.get('longitude', 0)
            traffic_volume = record.get('vehicle_count', 0)
            
            # Spatial analysis
            is_downtown = (40.7 < lat < 40.8) and (-74.1 < lon < -73.9)  # NYC example
            congestion_level = 'high' if traffic_volume > 100 else 'medium' if traffic_volume > 50 else 'low'
            
            traffic_analysis.append({
                **record,
                'is_downtown': is_downtown,
                'congestion_level': congestion_level,
                'traffic_density': traffic_volume / record.get('road_capacity', 1)
            })
        return traffic_analysis

analyzed_traffic = traffic_data.map_batches(TrafficAnalyzer(), batch_size=200)
```

### **Step 2: Event Log Analysis**

```python
# Process event logs (from log-ingestion template)
class EventLogProcessor:
    def __call__(self, batch):
        processed_events = []
        for log_entry in batch:
            event_type = log_entry.get('event_type', 'unknown')
            timestamp = log_entry.get('timestamp', '')
            location = log_entry.get('location', {})
            
            processed_event = {
                **log_entry,
                'is_traffic_event': event_type in ['accident', 'construction', 'road_closure'],
                'is_peak_hour': 7 <= int(timestamp[11:13]) <= 9 or 17 <= int(timestamp[11:13]) <= 19,
                'event_severity': 'high' if event_type == 'accident' else 'medium'
            }
            processed_events.append(processed_event)
        return processed_events

processed_events = event_logs.map_batches(EventLogProcessor(), batch_size=500)
```

### **Step 3: Time Series Trend Analysis**

```python
# Forecast traffic trends (from time-series-forecasting template)
class TrafficForecaster:
    def __call__(self, batch):
        import pandas as pd
        
        df = pd.DataFrame(batch)
        if df.empty:
            return []
        
        # Simple trend analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_traffic = df.groupby('hour')['vehicle_count'].mean()
        
        forecasts = []
        for hour, avg_traffic in hourly_traffic.items():
            # Simple forecast (in practice, use sophisticated models)
            forecast = {
                'hour': hour,
                'current_avg_traffic': avg_traffic,
                'predicted_traffic': avg_traffic * (1 + np.random.uniform(-0.1, 0.1)),
                'confidence_interval': [avg_traffic * 0.9, avg_traffic * 1.1]
            }
            forecasts.append(forecast)
        
        return forecasts

# Generate traffic forecasts
traffic_forecasts = analyzed_traffic.map_batches(TrafficForecaster(), batch_size=1000)

# Combine insights
print("Urban Planning Analysis Complete:")
print(f"  Traffic patterns analyzed: {analyzed_traffic.count()} records")
print(f"  Events processed: {processed_events.count()} events")  
print(f"  Hourly forecasts: {traffic_forecasts.count()} predictions")
```

## Workflow 4: Document Intelligence Platform

**Use Case**: Enterprise document processing with quality assurance

**Templates Used**: Unstructured Data Ingestion → Data Quality Monitoring → Enterprise Data Catalog

### **Step 1: Document Processing**

```python
# Process documents (from unstructured-data-ingestion template)
from ray.data import read_binary_files

# Load enterprise documents
documents = read_binary_files("s3://enterprise-docs/", include_paths=True)

class DocumentProcessor:
    def __call__(self, batch):
        processed_docs = []
        for doc in batch:
            # Extract text and metadata (simplified)
            file_path = doc.get('path', '')
            file_size = len(doc.get('bytes', b''))
            
            # Simulate document processing
            extracted_text = f"Extracted text from {file_path} ({file_size} bytes)"
            
            processed_doc = {
                'document_id': hash(file_path) % 10000,
                'file_path': file_path,
                'extracted_text': extracted_text,
                'file_size': file_size,
                'document_type': 'pdf' if '.pdf' in file_path else 'other',
                'processing_status': 'success'
            }
            processed_docs.append(processed_doc)
        return processed_docs

processed_documents = documents.map_batches(DocumentProcessor(), batch_size=10)
```

### **Step 2: Document Quality Assessment**

```python
# Apply quality monitoring (from data-quality-monitoring template)
class DocumentQualityValidator:
    def __call__(self, batch):
        quality_results = []
        for doc in batch:
            # Validate document quality
            text_length = len(doc.get('extracted_text', ''))
            file_size = doc.get('file_size', 0)
            
            quality_score = 1.0
            if text_length < 100:
                quality_score -= 0.3  # Too short
            if file_size > 10 * 1024 * 1024:
                quality_score -= 0.2  # Very large file
            
            quality_results.append({
                **doc,
                'quality_score': quality_score,
                'text_length': text_length,
                'is_processable': quality_score >= 0.6
            })
        return quality_results

quality_assessed = processed_documents.map_batches(DocumentQualityValidator(), batch_size=20)
processable_docs = quality_assessed.filter(lambda x: x['is_processable'])
```

### **Step 3: Catalog and Governance**

```python
# Catalog documents (from enterprise-data-catalog template)
class DocumentCatalogManager:
    def __call__(self, batch):
        cataloged_docs = []
        for doc in batch:
            catalog_entry = {
                **doc,
                'catalog_id': f"DOC_{doc['document_id']}",
                'data_classification': 'internal',
                'retention_period': '7_years',
                'access_controls': ['legal_team', 'compliance_team'],
                'processing_lineage': 'raw_docs → extraction → quality_check → catalog',
                'governance_status': 'compliant'
            }
            cataloged_docs.append(catalog_entry)
        return cataloged_docs

final_catalog = processable_docs.map_batches(DocumentCatalogManager(), batch_size=50)

print("Document Intelligence Platform Results:")
print(f"  Documents processed: {processed_documents.count()}")
print(f"  Quality-approved documents: {processable_docs.count()}")
print(f"  Cataloged documents: {final_catalog.count()}")
```

## Workflow 5: Real-Time Analytics Platform

**Use Case**: Operational monitoring combining logs, metrics, and forecasting

**Templates Used**: Log Ingestion → Time Series Forecasting → Data Quality Monitoring

### **Step 1: Log Processing**

```python
# Process operational logs (from log-ingestion template)
from ray.data import read_text

# Load application logs
app_logs = read_text("s3://ops-logs/application/")

class OperationalLogProcessor:
    def __call__(self, batch):
        metrics = []
        for log_entry in batch:
            # Parse log entry (simplified)
            text = log_entry.get('text', '')
            
            # Extract operational metrics
            if 'ERROR' in text:
                metric_type = 'error'
                severity = 'high'
            elif 'WARN' in text:
                metric_type = 'warning'
                severity = 'medium'
            else:
                metric_type = 'info'
                severity = 'low'
            
            metrics.append({
                'timestamp': datetime.now().isoformat(),
                'metric_type': metric_type,
                'severity': severity,
                'log_source': 'application',
                'requires_attention': severity in ['high', 'medium']
            })
        return metrics

operational_metrics = app_logs.map_batches(OperationalLogProcessor(), batch_size=100)
```

### **Step 2: Trend Forecasting**

```python
# Forecast operational trends (from time-series-forecasting template)
class OperationalForecaster:
    def __call__(self, batch):
        import pandas as pd
        
        df = pd.DataFrame(batch)
        if df.empty:
            return []
        
        # Analyze error rate trends
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        error_rates = df.groupby('hour')['metric_type'].apply(
            lambda x: (x == 'error').sum() / len(x)
        )
        
        forecasts = []
        for hour, error_rate in error_rates.items():
            forecast = {
                'hour': hour,
                'current_error_rate': error_rate,
                'predicted_error_rate': error_rate * (1 + np.random.uniform(-0.2, 0.2)),
                'alert_threshold': 0.05,
                'needs_attention': error_rate > 0.05
            }
            forecasts.append(forecast)
        
        return forecasts

trend_forecasts = operational_metrics.map_batches(OperationalForecaster(), batch_size=500)
```

### **Step 3: Quality Monitoring**

```python
# Monitor operational quality (from data-quality-monitoring template)
class OperationalQualityMonitor:
    def __call__(self, batch):
        quality_metrics = []
        for forecast in batch:
            # Assess operational quality
            error_rate = forecast.get('current_error_rate', 0)
            predicted_rate = forecast.get('predicted_error_rate', 0)
            
            quality_score = 1.0 - error_rate  # Higher error rate = lower quality
            trend_direction = 'improving' if predicted_rate < error_rate else 'degrading'
            
            quality_metrics.append({
                **forecast,
                'operational_quality_score': quality_score,
                'trend_direction': trend_direction,
                'system_health': 'good' if quality_score > 0.9 else 'warning' if quality_score > 0.8 else 'critical'
            })
        return quality_metrics

quality_monitoring = trend_forecasts.map_batches(OperationalQualityMonitor(), batch_size=100)

# Generate operational dashboard data
dashboard_data = quality_monitoring.take_all()
print("Operational Analytics Platform Results:")
print(f"  Log entries processed: {operational_metrics.count()}")
print(f"  Trend forecasts generated: {trend_forecasts.count()}")
print(f"  Quality assessments: {len(dashboard_data)}")
```

## Best Practices for Cross-Template Workflows

### **Data Flow Optimization**

```python
# Optimize data flow between templates
def optimize_cross_template_pipeline(datasets):
    """Optimize data flow across multiple template operations."""
    
    # 1. Minimize data movement
    # Process related operations together
    combined_result = dataset.map_batches(
        lambda batch: template1_process(template2_process(batch)),
        batch_size=optimal_batch_size
    )
    
    # 2. Use appropriate data formats
    # Save intermediate results in efficient formats
    intermediate_data.write_parquet("local://temp/intermediate_results")
    
    # 3. Leverage Ray Data's lazy evaluation
    # Chain operations before execution
    final_result = (dataset
                   .map_batches(step1_processor)
                   .filter(quality_filter)
                   .map_batches(step2_processor)
                   .groupby('category').count())
    
    return final_result
```

### **Resource Management**

```python
# Manage resources across workflow stages
def configure_workflow_resources():
    """Configure optimal resources for cross-template workflows."""
    
    resources = ray.cluster_resources()
    
    # Adjust concurrency based on workflow stage
    if 'GPU' in resources and resources['GPU'] > 0:
        # GPU-intensive stages (AI/ML processing)
        gpu_config = {'batch_size': 16, 'concurrency': 2, 'num_gpus': 1}
        
        # CPU-intensive stages (data processing)
        cpu_config = {'batch_size': 100, 'concurrency': 8, 'num_gpus': 0}
    else:
        # CPU-only configuration
        cpu_config = {'batch_size': 50, 'concurrency': 4, 'num_gpus': 0}
        gpu_config = cpu_config
    
    return {'gpu_config': gpu_config, 'cpu_config': cpu_config}
```

### **Error Handling Across Templates**

```python
# Implement robust error handling for multi-template workflows
def robust_workflow_execution(workflow_steps):
    """Execute multi-template workflow with comprehensive error handling."""
    
    results = {}
    
    for step_name, step_function in workflow_steps.items():
        try:
            logger.info(f"Executing workflow step: {step_name}")
            step_result = step_function()
            results[step_name] = step_result
            
            # Validate step output
            if hasattr(step_result, 'count'):
                count = step_result.count()
                logger.info(f"Step {step_name} completed: {count} records")
                
                if count == 0:
                    logger.warning(f"Step {step_name} produced no results")
            
        except Exception as step_error:
            logger.error(f"Step {step_name} failed: {step_error}")
            
            # Decide whether to continue or abort
            if step_name in ['data_loading', 'critical_validation']:
                logger.error("Critical step failed, aborting workflow")
                raise
            else:
                logger.warning(f"Non-critical step {step_name} failed, continuing...")
                results[step_name] = None
    
    return results
```

## Summary

These cross-template workflows demonstrate how Ray Data templates can be combined to create comprehensive data processing solutions. Each workflow leverages the strengths of multiple templates while maintaining the simplicity and efficiency of Ray Data's native operations.

**Key Benefits of Cross-Template Workflows:**
- **Comprehensive Coverage**: Address complex business scenarios
- **Reusable Components**: Leverage existing template implementations  
- **Scalable Architecture**: Built on Ray Data's distributed processing
- **Production Ready**: Include error handling and monitoring
- **Anyscale Optimized**: Designed for managed Ray clusters
