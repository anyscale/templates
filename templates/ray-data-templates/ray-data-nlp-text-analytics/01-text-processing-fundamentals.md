# Part 1: Text Processing Fundamentals

**⏱️ Time to complete**: 15 min

**[← Back to Overview](README.md)** | **[Continue to Part 2 →](02-nlp-analysis-insights.md)**

---

## Learning Objectives

**What you'll learn:**
- Load text data from various sources using Ray Data
- Preprocess and clean text at scale with distributed operations
- Create interactive text analytics visualizations with plotly and word clouds
- Use Ray Data's `map_batches()` and native operations for text workflows

**Why this matters:**
- **Text processing fundamentals**: Learn distributed text loading, cleaning, and tokenization
- **Ray Data operations**: Master `map_batches()`, `filter()`, and `groupby()` for text data
- **Production pipelines**: Build scalable text preprocessing pipelines
- **Real-world applications**: Process millions of documents like e-commerce reviews and social media posts

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites-checklist)
3. [Quick Start](#quick-start-3-minutes)
4. [Loading Text Data](#step-1-loading-text-data)
5. [Interactive Visualizations](#interactive-text-analytics-visualizations)

---

## Overview

### The Challenge

Processing large text datasets (reviews, social media, documents) faces these limitations:
- **Memory constraints**: Traditional tools load entire datasets into RAM
- **Single-core processing**: Sequential text operations are slow
- **Scalability issues**: Can't handle millions of documents efficiently
- **Resource bottlenecks**: Text preprocessing becomes the bottleneck

### The Solution

Ray Data enables distributed text processing:
- **Parallel processing**: Distribute text operations across multiple cores
- **Memory efficiency**: Stream processing handles unlimited text volumes
- **Native operations**: Use `map_batches()` for distributed text transformations
- **Scalability**: Analyze millions of documents quickly

### Real-World Impact

**Production text processing use cases:**

| Industry | Use Case | Scale | Benefit |
|----------|----------|-------|---------|
| **E-commerce** | Product review analysis | Millions of reviews | Customer insights |
| **Social Media** | Post sentiment analysis | Billions of posts | Trending topics |
| **News** | Article classification | 100K+ articles/day | Content categorization |
| **Customer Support** | Ticket routing | Millions of tickets | Automated triage |

---

## Prerequisites

**Before starting, ensure you have:**
- [ ] Basic understanding of text processing concepts (tokenization, cleaning)
- [ ] Familiarity with NLP terminology
- [ ] Python environment with 4GB+ RAM recommended
- [ ] Understanding of machine learning basics

---

## Quick Start (3 minutes)

This section demonstrates text processing using Ray Data:

```python
import ray

# Initialize Ray for distributed processing
ray.init()

# Create sample text data
texts = ["I love this product", "This is terrible", "Pretty good overall"]
text_dataset = ray.data.from_items([{"text": t} for t in texts * 1000])

print(f"Created dataset with {text_dataset.count():,} text samples")
print("Ray Data text processing ready!")
```

**Installation:**

```bash
# Install all required dependencies
pip install ray[data] transformers torch nltk wordcloud matplotlib seaborn plotly textstat
```

---

---

## Step 1: Loading Text Data

*Time: 5 minutes*

### What You'll Build

Create a realistic text dataset similar to product reviews or social media posts for meaningful analysis without requiring huge downloads.

### Why Realistic Data Matters

**Learning with production-like data:**
- Text patterns match real-world use cases
- Techniques scale naturally from thousands to millions of documents
- No architectural changes needed when moving to production
- Development insights translate directly to production environments

**Memory efficiency benefits:**
- Traditional methods load entire datasets into RAM (bottleneck)
- Ray Data's distributed approach processes unlimited text volumes
- Stream processing prevents memory constraints
- Analyze complete text archives instead of samples

```python
# Demonstrate scalable text processing efficiency
def measure_text_processing_efficiency():
    """Show how Ray Data handles increasing text volumes."""
    # Start with smaller dataset for comparison
    small_texts = ["Sample text"] * 1000
    small_dataset = ray.data.from_items([{"text": t} for t in small_texts])
    
    # Scale to larger dataset 
    large_texts = ["Sample text"] * 100000
    large_dataset = ray.data.from_items([{"text": t} for t in large_texts])
    
    print(f"Small dataset: {small_dataset.count():,} texts")
    print(f"Large dataset: {large_dataset.count():,} texts")
    print("Memory usage remains constant - Ray Data streams processing")
    
    return large_dataset

# Demonstrate memory-efficient text processing
efficient_dataset = measure_text_processing_efficiency()
```

**Key benefit:** This scalable foundation enables efficient text analytics that work consistently across different data volumes and organizational requirements.

```python
import ray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import textstat
from collections import Counter
import re

# Initialize Ray for distributed processing
ray.init()

def load_real_text_data():
    """Load real Amazon product reviews for text analytics."""
    print("Loading real Amazon product review dataset...")
    
    # Load real Amazon product reviews from public dataset
    try:
        # Load Amazon reviews parquet data
        text_dataset = ray.data.read_parquet(
            "s3://amazon-reviews-pds/parquet/product_category=Books/",
            columns=["review_body", "star_rating", "product_title", "verified_purchase"],
            num_cpus=0.025
        ).limit(10000)  # Limit to 10K reviews for processing efficiency
        
        print(f"Loaded {text_dataset.count():,} Amazon book reviews")
        
        # BEST PRACTICE: Use Ray Data native operations for data transformation
        from ray.data.expressions import col, lit
        
        # Add sentiment mapping using native column operations
        def map_sentiment(batch):
            """Map star ratings to sentiment labels efficiently."""
            transformed = []
            for review in batch:
                # Map star rating to sentiment
                rating = review.get('star_rating', 3)
                if rating >= 4:
                    sentiment = 'positive'
                elif rating <= 2:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                # Calculate text length without pandas overhead
                text_length = len(str(review.get('review_body', '')))
                
                transformed.append({
                    'text': review.get('review_body', ''),
                    'sentiment': sentiment,
                    'star_rating': rating,
                    'product_title': review.get('product_title', ''),
                    'verified_purchase': review.get('verified_purchase', False),
                    'length': text_length
                })
            
            return transformed
        
        # Apply transformation with optimized batch processing
        text_dataset = text_dataset.map_batches(
            map_sentiment,
            batch_size=2000,  # Larger batch size for efficiency
            concurrency=4     # Parallel processing
        , batch_format="pandas")
        
        # Use native column operations for additional features
        text_dataset = text_dataset.add_column(
            "is_long_review", 
            col("length") > lit(500)
        ).add_column(
            "is_positive",
            col("star_rating") >= lit(4)
        )
        
        return text_dataset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Amazon reviews: {e}")

# Load real text datasettext_dataset = load_real_text_data()

# Display basic information about our datasetprint(f"Loaded dataset with {text_dataset.count():,} text samples")
print(f"Schema: {text_dataset.schema()}")

# Show a few sample textsprint("\nSample texts:")
samples = text_dataset.take(3)
for i, sample in enumerate(samples):
    text_preview = sample['text'][:100] + "..." if len(sample['text']) > 100 else sample['text']
    print(f"{i+1}. {text_preview} (sentiment: {sample.get('sentiment', 'unknown')})")
```

### Interactive NLP Text Analytics Dashboard

```python
# Create an engaging NLP text analytics visualization dashboard
def create_nlp_dashboard(dataset, sample_size=1000):
    """Generate a comprehensive NLP text analytics dashboard."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import Counter
    import re
    from wordcloud import WordCloud
    
    # Sample data for analysis
    sample_data = dataset.take(sample_size)
    df = pd.DataFrame(sample_data)
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    # 1. Sentiment Distribution
    ax_sentiment = fig.add_subplot(gs[0, :2])
    sentiment_counts = df['true_sentiment'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax_sentiment.pie(sentiment_counts.values, 
                                               labels=sentiment_counts.index, 
                                               autopct='%1.1f%%', colors=colors, 
                                               startangle=90)
    ax_sentiment.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # 2. Text Length Analysis
    ax_length = fig.add_subplot(gs[0, 2:])
    
    # Data transformation
    df['length'] = df['text'].str.len()
    ax_length.hist(df['length'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax_length.axvline(df['length'].mean(), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {df["length"].mean():.1f} chars')
    ax_length.set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    ax_length.set_xlabel('Character Count')
    ax_length.set_ylabel('Frequency')
    ax_length.legend()
    ax_length.grid(True, alpha=0.3)
    
    # Simplified text analysis
    
    # Data transformation
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\nText Analytics Summary:")
    print(f"Total reviews: {len(df):,}")
    print(f"Average text length: {df['length'].mean():.1f} characters")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    
    # Sentiment distribution
    sentiment_dist = df['true_sentiment'].value_counts()
    print(f"\nSentiment distribution:")
    for sentiment, count in sentiment_dist.items():
        print(f"  {sentiment}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 4. Sentiment vs Length Analysis
    ax_sent_length = fig.add_subplot(gs[1, 2:])
    for sentiment in df['true_sentiment'].unique():
        sentiment_data = df[df['true_sentiment'] == sentiment]
        ax_sent_length.scatter(sentiment_data['length'], sentiment_data['word_count'], 
                              label=sentiment, alpha=0.6, s=30)
    ax_sent_length.set_title('Sentiment vs Text Characteristics', fontsize=12, fontweight='bold')
    ax_sent_length.set_xlabel('Character Count')
    ax_sent_length.set_ylabel('Word Count')
    ax_sent_length.legend()
    ax_sent_length.grid(True, alpha=0.3)
    
    # 5. Most Common Words
    ax_words_common = fig.add_subplot(gs[2, :2])
    all_text = ' '.join(df['text'].str.lower())
    # Remove common stop words and punctuation
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_counts = Counter(words)
    common_words = dict(word_counts.most_common(15))
    
    words_list = list(common_words.keys())
    counts_list = list(common_words.values())
    
    bars = ax_words_common.barh(range(len(words_list)), counts_list, 
                               color=plt.cm.viridis(np.linspace(0, 1, len(words_list))))
    ax_words_common.set_yticks(range(len(words_list)))
    ax_words_common.set_yticklabels(words_list)
    ax_words_common.set_title('Most Common Words', fontsize=12, fontweight='bold')
    ax_words_common.set_xlabel('Frequency')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts_list)):
        width = bar.get_width()
        ax_words_common.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                            f'{count}', ha='left', va='center', fontsize=9)
    
    # 6. Sentiment by Word Count
    ax_sent_words = fig.add_subplot(gs[2, 2:])
    sentiment_word_avg = df.groupby('true_sentiment')['word_count'].mean()
    bars = ax_sent_words.bar(range(len(sentiment_word_avg)), sentiment_word_avg.values,
                            color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax_sent_words.set_title('Average Word Count by Sentiment', fontsize=12, fontweight='bold')
    ax_sent_words.set_ylabel('Average Word Count')
    ax_sent_words.set_xticks(range(len(sentiment_word_avg)))
    ax_sent_words.set_xticklabels(sentiment_word_avg.index)
    
    # Add average labels
    for bar, avg in zip(bars, sentiment_word_avg.values):
        height = bar.get_height()
        ax_sent_words.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Text Statistics Summary
    ax_stats = fig.add_subplot(gs[3, :2])
    ax_stats.axis('off')
    
    # Calculate text statistics
    total_reviews = len(df)
    avg_length = df['length'].mean()
    avg_words = df['word_count'].mean()
    unique_words = len(set(' '.join(df['text']).lower().split()))
    sentiment_dist = df['true_sentiment'].value_counts()
    
    stats_text = "Text Analytics Summary\n" + "="*50 + "\n"
    stats_text += f"Total Reviews: {total_reviews:,}\n"
    stats_text += f"Average Length: {avg_length:.1f} characters\n"
    stats_text += f"Average Words: {avg_words:.1f} words\n"
    stats_text += f"Unique Words: {unique_words:,}\n"
    stats_text += f"Vocabulary Density: {unique_words/len(' '.join(df['text']).split()):.3f}\n"
    stats_text += f"Positive: {sentiment_dist.get('positive', 0):,} ({sentiment_dist.get('positive', 0)/total_reviews*100:.1f}%)\n"
    stats_text += f"Negative: {sentiment_dist.get('negative', 0):,} ({sentiment_dist.get('negative', 0)/total_reviews*100:.1f}%)\n"
    stats_text += f"Neutral: {sentiment_dist.get('neutral', 0):,} ({sentiment_dist.get('neutral', 0)/total_reviews*100:.1f}%)\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # 8. Sample Reviews Table
    ax_table = fig.add_subplot(gs[3, 2:])
    ax_table.axis('off')
    
    # Create sample reviews table
    sample_df = df.head(6)[['review_id', 'true_sentiment', 'text', 'word_count']].copy()
    sample_df['text'] = sample_df['text'].str[:40] + '...'  # Truncate long texts
    
    table_text = "Sample Reviews\n" + "="*80 + "\n"
    table_text += f"{'ID':<12} {'Sentiment':<10} {'Text':<30} {'Words':<6}\n"
    table_text += "-"*80 + "\n"
    
    for _, row in sample_df.iterrows():
        table_text += f"{row['review_id']:<12} {row['true_sentiment']:<10} {row['text']:<30} {row['word_count']:<6}\n"
    
    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('NLP Text Analytics Dashboard', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    print(plt.limit(10).to_pandas())
    
    # Print NLP insights
    print("NLP Text Analytics Insights:")
    print(f"- Text diversity: {unique_words:,} unique words across {total_reviews:,} reviews")
    print(f"- Average complexity: {avg_words:.1f} words per review")
    print(f"- Sentiment balance: {sentiment_dist.to_dict()}")
    print(f"- Vocabulary density: {unique_words/len(' '.join(df['text']).split()):.3f}")
    
    return df

# Generate the NLP dashboardnlp_df = create_nlp_dashboard(text_dataset)
```

**Why This Dashboard Matters:**
- **Text Understanding**: Visualize text characteristics and patterns across different sentiments
- **Quality Assessment**: Analyze text length, word count, and vocabulary diversity
- **Sentiment Analysis**: Understand sentiment distribution and text characteristics
- **Pattern Recognition**: Identify common words and linguistic patterns in the dataset

** What just happened?**
- Created 1,000 realistic text samples (reviews)
- Each sample has text content and known sentiment
- Data is loaded into Ray Data for distributed processing
- We can easily scale this to millions of real reviews

## Use Case: Enterprise Content Intelligence Platform

### Real-world Business Scenario

A large e-commerce company receives 100,000+ pieces of text content daily from multiple sources and needs to extract actionable business insights with large datasets. Traditional NLP tools can't handle this volume efficiently.

**Content Sources and Volumes:**
- **Customer Reviews**: 50,000+ daily product and service reviews
- **Support Tickets**: 15,000+ daily customer service interactions  
- **Social Media**: 25,000+ daily mentions, posts, and comments
- **Internal Documents**: 10,000+ daily emails, reports, and documentation

**Business Challenges:**
- **Manual Processing**: Takes 40+ hours daily with human analysts
- **Inconsistent Analysis**: Different analysts provide varying insights
- **Delayed Response**: 24-48 hour delay for sentiment analysis and issue identification
- **Limited Scale**: Can only process 10% of available content
- **High Cost**: Expensive external NLP service dependencies

### Ray Data Solution Benefits

The comprehensive NLP pipeline delivers:

| Business Aspect | Traditional Approach | Ray Data Solution | Business Impact |
|----------------|----------------------|-------------------|-----------------|
| **Processing Speed** | Manual analysis required | Automated distributed processing | Faster insights |
| **Content Coverage** | Sample-based analysis | Full-dataset processing capability | Complete coverage |
| **Consistency** | Variable analyst results | Standardized ML processing | Reliable outcomes |
| **Scalability** | Limited to analyst capacity | Distributed across workers | Horizontal scaling |
| **Cost Structure** | External service dependencies | Infrastructure automation | Operational efficiency |
| **Response Time** | Significant processing delays | Near real-time processing | Faster decision making |

### Enterprise NLP Pipeline Capabilities

The pipeline provides comprehensive text intelligence:

1. **Content Classification and Routing**
   - Automatically categorize incoming content by type and urgency
   - Route high-priority issues to appropriate teams
   - Identify trending topics and emerging issues

2. **Customer Experience Analytics**
   - Real-time sentiment monitoring across all channels
   - Product satisfaction scoring and trend analysis
   - Customer pain point identification and escalation

3. **Competitive Intelligence**
   - Brand mention analysis and competitive comparison
   - Market sentiment tracking and trend identification
   - Product feature feedback and improvement suggestions

4. **Operational Efficiency**
   - Automated content summarization for executive reports
   - Key entity extraction for CRM enrichment
   - Multi-language content processing for global operations

## Architecture

### Ray Data NLP Processing Architecture

```
Enterprise Text Sources (100K+ daily)
 Customer Reviews (50K)
 Support Tickets (15K) 
 Social Media (25K)
 Documents (10K)
         
         

           Ray Data Ingestion            
   read_text()  read_parquet(, num_cpus=0.025)        
   from_huggingface()  read_json(, num_cpus=0.05)    
   Distributed loading across cluster  

                  
                  

     Distributed Text Processing         
   map_batches() for vectorized ops    
   Parallel preprocessing across nodes 
   Memory-efficient text cleaning      
   Automatic load balancing           

                  
                  

        Multi-Model NLP Analysis         
   BERT embeddings (GPU accelerated)   
   Sentiment analysis (transformer)    
   Topic modeling (LDA + clustering)   
   Named entity recognition (spaCy)    
   Text summarization (BART)          
   Language detection (multilingual)   

                  
                  

      Ray Data LLM Integration           (Optional)
   Production LLM inference           
   Batch processing optimization      
   Structured prompt engineering      
   GPU resource management           

                  
                  

      Business Intelligence Layer        
   Aggregated insights and metrics     
   Interactive dashboards             
   Real-time alerts and notifications 
   Executive reporting and analytics  

```

### Ray Data Advantages for NLP

| Traditional NLP Approach | Ray Data NLP Approach | Business Impact |
|---------------------------|----------------------|-----------------|
| **Single-machine processing** | Distributed across 88+ CPU cores | scale increase |
| **Sequential model inference** | Parallel GPU acceleration | faster processing |
| **Manual pipeline orchestration** | Native Ray Data operations | 80% less infrastructure code |
| **Complex resource management** | Automatic scaling and load balancing | Zero ops overhead |
| **Limited fault tolerance** | Built-in error recovery and retries | 99.9% pipeline reliability |

## Key Components

### 1. Text Data Loading
- `ray.data.read_text()` for text files
- `ray.data.read_parquet(, num_cpus=0.025)` for structured text data
- Custom readers for specific text formats
- Text data validation and schema management

### 2. Text Preprocessing
- Text cleaning and normalization
- Tokenization and stemming
- Stop word removal and lemmatization
- Language detection and encoding handling

### 3. NLP Model Integration
- Pre-trained language models (BERT, RoBERTa, GPT)
- Custom model training and fine-tuning
- Embedding generation and similarity analysis
- Multi-language support and localization

### 4. Text Analytics
- Sentiment analysis and emotion detection
- Topic modeling and clustering
- Named entity recognition
- Text classification and categorization

## Prerequisites

- Ray cluster with GPU support (recommended)
- Python 3.8+ with NLP libraries
- Access to text datasets
- Basic understanding of NLP concepts and techniques

## Installation

```bash
pip install ray[data] transformers torch
pip install nltk spacy textblob
pip install sentence-transformers scikit-learn
pip install pandas numpy pyarrow
```

## 5-Minute Quick Start

**Goal**: Analyze sentiment of real text data in 5 minutes

### Step 1: Setup on Anyscale (30 Seconds)

```python
# Ray cluster is already running on Anyscale
import ray

# Check cluster status (already connected)
print('Connected to Anyscale Ray cluster')
print(f'Available resources: {ray.cluster_resources()}')

# Install any missing packages if needed
# !pip install transformers torch
```

### Step 2: Load Real Text Data (1 Minute)

```python
import ray

# Create sample real movie reviews for quick demoreal_reviews = [
    "This movie was absolutely excellent! Great acting and plot.",
    "Terrible film. Waste of time and money. Very disappointed.",
    "Amazing cinematography and outstanding performances throughout.",
    "The movie was okay, nothing special but entertaining enough.",
    "Brilliant storytelling and efficient attention to detail."
]

# Convert to Ray datasettext_ds = ray.data.from_items([{"text": review, "id": i} for i, review in enumerate(real_reviews)])
print(f"Loaded {text_ds.count()} real movie reviews")
```

### Step 3: Run Sentiment Analysis (2 Minutes)

```python
from transformers import pipeline

class QuickSentimentAnalyzer:
    def __init__(self):

    """  Init  ."""

    """  Init  ."""

    """  Init  ."""
        self.sentiment_pipeline = pipeline("sentiment-analysis", device=-1)  # CPU for speed
    
    def __call__(self, batch):

    """  Call  ."""
        results = []
        for item in batch:
            try:
                text = item["text"]
                sentiment = self.sentiment_pipeline(text[:512])[0]
                results.append({
                    **item,
                    "sentiment": sentiment["label"],
                    "confidence": sentiment["score"]
                })
            except Exception as e:
                results.append({**item, "error": str(e)})
        return results

# Analyze sentiment
sentiment_results = text_ds.map_batches(
    QuickSentimentAnalyzer,
    num_cpus=0.25,
    batch_size=5
)
final_results = sentiment_results.take_all()
```

### Step 4: View Results (1 Minute)

```python
# Display sentiment analysis results
print("\nSentiment Analysis Results:")
print("-" * 50)

for result in final_results:
    text = result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
    sentiment = result.get("sentiment", "ERROR")
    confidence = result.get("confidence", 0)
    
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print("-" * 30)
```

## Interactive Text Analytics Visualizations

Create stunning visualizations to analyze our text data:

### Word Clouds and Text Analysis

```python
# Create engaging text analytics visualizations
from util.viz_utils import create_word_clouds, create_simple_sentiment_chart
import pandas as pd

# Convert data for visualization
text_df = dataset.to_pandas()
sentiment_df = pd.DataFrame(final_results)

# Create sentiment chart
sentiment_fig = create_simple_sentiment_chart(sentiment_df)
if sentiment_fig:
    sentiment_fig.show()

# Create word clouds for positive and negative sentiment
wordcloud_fig = create_word_clouds(sentiment_df)
wordcloud_fig.savefig('sentiment_wordclouds.png', dpi=150, bbox_inches='tight')

print("Text analytics visualizations created")
```

### Interactive Text Analytics Dashboard

```python
# Create interactive text analytics dashboard
from util.viz_utils import create_text_analytics_dashboard
import pandas as pd

# Convert data for visualization
text_df = dataset.to_pandas()
sentiment_df = pd.DataFrame(sentiment_results)

# Create comprehensive dashboard with interactive charts
dashboard_fig = create_text_analytics_dashboard(text_df, sentiment_df)
dashboard_fig.show()

# Print summary
print("\nText Analytics Summary:")
print(f"  Total texts analyzed: {len(text_df):,}")
print(f"  Average text length: {text_df['text'].str.len().mean():.0f} characters")
print(f"  Average word count: {text_df['text'].str.split().str.len().mean():.0f} words")
if 'sentiment' in sentiment_df.columns:
    print(f"  Sentiment breakdown: {sentiment_df['sentiment'].value_counts().to_dict()}")

print("Quick start completed! Run the full demo for improved NLP features.")

# Expected Output:# Sentiment Analysis Results:# --------------------------------------------------
# Text: This movie was absolutely excellent! Great acting...# Sentiment: POSITIVE (confidence: 0.95)# ------------------------------
# Text: Terrible film. Waste of time and money. Very dis...# Sentiment: NEGATIVE (confidence: 0.92)# ------------------------------
# Text: Amazing cinematography and outstanding performanc...# Sentiment: POSITIVE (confidence: 0.88)```

