# Part 2: NLP Analysis and Insights

**⏱️ Time to complete**: 15 min

**[← Back to Part 1](01-text-processing-fundamentals.md)** | **[Return to Overview](README.md)**

---

## What You'll Learn

In this part, you'll master advanced NLP analysis techniques:
1. Implement distributed sentiment analysis
2. Perform topic modeling and clustering
3. Extract named entities from text
4. Deploy NLP pipelines to production

## Prerequisites

Complete [Part 1: Text Processing Fundamentals](01-text-processing-fundamentals.md) before starting this part.

## Table of Contents

1. [Complete NLP Tutorial](#complete-nlp-tutorial)
2. [Advanced Features](#advanced-features)
3. [Production Considerations](#production-considerations)
4. [Example Workflows](#example-workflows)

---

## Complete NLP Tutorial

### 1. Load Real Text Data

```python
import ray
from ray.data import read_text, read_parquet, from_huggingface

# Ray cluster is already running on Anyscaleprint(f'Ray cluster resources: {ray.cluster_resources()}')

# Load real text datasets using Ray Data native APIs# Use Ray Data's native Hugging Face integrationimdb_reviews = from_huggingface("imdb", split="train[:1000]")
print(f"IMDB Reviews: {imdb_reviews.count()}")

# Load Amazon reviews using native read_parquetamazon_reviews = read_parquet(
    "s3://amazon-reviews-pds/parquet/product_category=Books/",
    columns=["review_body", "star_rating"]
).limit(500)
print(f"Amazon Reviews: {amazon_reviews.count()}")

# Inspect sample data
sample_review = imdb_reviews.take(1)[0]
print(f"Sample review: {sample_review['text'][:100]}...")
print(f"Sample label: {sample_review['label']}")
```

### 2. Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Preprocess text data for NLP analysis."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def __call__(self, batch):
        """Preprocess a batch of text documents."""
        processed_texts = []
        
        for text in batch["text"]:
            try:
                # Clean and normalize text
                cleaned_text = self._clean_text(text)
                
                # Tokenize
                tokens = self._tokenize(cleaned_text)
                
                # Remove stop words and lemmatize
                processed_tokens = self._process_tokens(tokens)
                
                # Join tokens back into text
                processed_text = " ".join(processed_tokens)
                
                processed_texts.append({
                    "original_text": text,
                    "processed_text": processed_text,
                    "token_count": len(processed_tokens),
                    "cleaned_length": len(processed_text)
                })
                
            except Exception as e:
                print(f"Error preprocessing text: {e}")
                continue
        
        return {"processed_texts": processed_texts}
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def _tokenize(self, text):
        """Tokenize text into words."""
        return nltk.word_tokenize(text)
    
    def _process_tokens(self, tokens):
        """Remove stop words and lemmatize tokens."""
        processed = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed.append(lemmatized)
        return processed

# Apply text preprocessingprocessed_texts = reviews_ds.map_batches(
    TextPreprocessor(, batch_format="pandas"),
    batch_size=100,
    concurrency=4
)
```

### 3. Sentiment Analysis

```python
from transformers import pipeline
import torch

class SentimentAnalyzer:
    """Perform sentiment analysis using pre-trained models."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load emotion detection pipeline
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def __call__(self, batch):
        """Analyze sentiment and emotions in text batch."""
        sentiment_results = []
        
        for item in batch["processed_texts"]:
            try:
                text = item["processed_text"]
                
                # Perform sentiment analysis
                sentiment_result = self.sentiment_pipeline(text[:512])[0]
                
                # Perform emotion detection
                emotion_result = self.emotion_pipeline(text[:512])[0]
                
                # Combine results
                result = {
                    "id": item.get("id"),
                    "original_text": item["original_text"],
                    "processed_text": text,
                    "sentiment": sentiment_result["label"],
                    "sentiment_score": sentiment_result["score"],
                    "emotion": emotion_result["label"],
                    "emotion_score": emotion_result["score"],
                    "token_count": item["token_count"]
                }
                
                sentiment_results.append(result)
                
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                continue
        
        return {"sentiment_results": sentiment_results}

# Apply sentiment analysissentiment_analysis = processed_texts.map_batches(
    SentimentAnalyzer(, batch_format="pandas"),
    batch_size=32,
    num_gpus=1 if ray.cluster_resources().get("GPU", 0) > 0 else 0,
    concurrency=2
)
```

### 4. Topic Modeling and Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np

class TopicModeler:
    """Perform topic modeling and text clustering."""
    
    def __init__(self, num_topics=10, num_clusters=5):

    """  Init  ."""
        self.num_topics = num_topics
        self.num_clusters = num_clusters
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize topic model
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )
        
        # Initialize clustering model
        self.kmeans_model = KMeans(
            n_clusters=num_clusters,
            random_state=42
        )
    
    def __call__(self, batch):
        """Perform topic modeling and clustering on text batch."""
        try:
            texts = [item["processed_text"] for item in batch["sentiment_results"]]
            
            if not texts:
                return {"topic_results": []}
            
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Topic modeling
            topic_distributions = self.lda_model.fit_transform(tfidf_matrix)
            
            # Clustering
            cluster_labels = self.kmeans_model.fit_predict(tfidf_matrix)
            
            # Get feature names for topic interpretation
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Extract top words for each topic
            top_words_per_topic = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_words_per_topic.append(top_words)
            
            # Combine results
            topic_results = []
            for i, item in enumerate(batch["sentiment_results"]):
                result = {
                    **item,
                    "topic_distribution": topic_distributions[i].tolist(),
                    "dominant_topic": int(np.argmax(topic_distributions[i])),
                    "topic_confidence": float(np.max(topic_distributions[i])),
                    "cluster_label": int(cluster_labels[i]),
                    "top_topic_words": top_words_per_topic[np.argmax(topic_distributions[i])]
                }
                topic_results.append(result)
            
            return {"topic_results": topic_results}
            
        except Exception as e:
            print(f"Error in topic modeling: {e}")
            return {"topic_results": []}

# Apply topic modelingtopic_modeling = sentiment_analysis.map_batches(
    TopicModeler(num_topics=8, num_clusters=4, batch_format="pandas"),
    batch_size=100,
    concurrency=2
)
```

### 5. Named Entity Recognition

```python
import spacy
from collections import defaultdict

class NERExtractor:
    """Extract named entities from text."""
    
    def __init__(self):
        # Load English language model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Download if not available
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def __call__(self, batch):
        """Extract named entities from text batch."""
        ner_results = []
        
        for item in batch["topic_results"]:
            try:
                text = item["processed_text"]
                
                # Process text with spaCy
                doc = self.nlp(text)
                
                # Extract entities by type
                entities = defaultdict(list)
                for ent in doc.ents:
                    entities[ent.label_].append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": ent.label_
                    })
                
                # Count entities
                entity_counts = {label: len(ents) for label, ents in entities.items()}
                
                # Add NER results
                result = {
                    **item,
                    "entities": dict(entities),
                    "entity_counts": entity_counts,
                    "total_entities": sum(entity_counts.values())
                }
                
                ner_results.append(result)
                
            except Exception as e:
                print(f"Error in NER extraction: {e}")
                continue
        
        return {"ner_results": ner_results}

# Apply NER extractionner_extraction = topic_modeling.map_batches(
    NERExtractor(, batch_format="pandas"),
    batch_size=50,
    concurrency=2
)
```

### 6. Advanced NLP Functions

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class AdvancedNLPProcessor:
    """Perform improved NLP tasks including summarization, language detection, and classification."""
    
    def __init__(self):
        # Initialize multiple NLP pipelines
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=-1)
        
        # Question answering pipeline
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
    
    def __call__(self, batch):
        """Apply improved NLP processing to text batch."""
        improved_results = []
        
        for item in batch["ner_results"]:
            try:
                text = item["processed_text"]
                original_text = item["original_text"]
                
                # Text summarization (for longer texts)
                summary = ""
                if len(original_text) > 500:
                    try:
                        summary_result = self.summarizer(original_text[:1024], max_length=150, min_length=30, do_sample=False)
                        summary = summary_result[0]['summary_text']
                    except Exception as e:
                        summary = f"Summarization failed: {str(e)}"
                
                # Language detection
                try:
                    language_result = self.language_detector(text[:512])
                    detected_language = language_result[0]['label']
                    language_confidence = language_result[0]['score']
                except Exception as e:
                    detected_language = "unknown"
                    language_confidence = 0.0
                
                # Text classification (additional classification beyond sentiment)
                try:
                    classification_result = self.classifier(text[:512])
                    text_category = classification_result[0]['label']
                    category_confidence = classification_result[0]['score']
                except Exception as e:
                    text_category = "unknown"
                    category_confidence = 0.0
                
                # Question answering (example questions)
                qa_results = []
                sample_questions = [
                    "What is the main topic?",
                    "What is the sentiment?",
                    "Who is mentioned?"
                ]
                
                for question in sample_questions:
                    try:
                        qa_result = self.qa_pipeline(question=question, context=original_text[:512])
                        qa_results.append({
                            "question": question,
                            "answer": qa_result['answer'],
                            "confidence": qa_result['score']
                        })
                    except Exception as e:
                        qa_results.append({
                            "question": question,
                            "answer": "N/A",
                            "confidence": 0.0
                        })
                
                # Text readability and complexity metrics
                word_count = len(text.split())
                sentence_count = len([s for s in original_text.split('.') if s.strip()])
                avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
                
                improved_result = {
                    **item,
                    "summary": summary,
                    "detected_language": detected_language,
                    "language_confidence": language_confidence,
                    "text_category": text_category,
                    "category_confidence": category_confidence,
                    "qa_results": qa_results,
                    "readability_metrics": {
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "avg_word_length": avg_word_length,
                        "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0
                    },
                    "improved_processing_timestamp": pd.Timestamp.now().isoformat()
                }
                
                improved_results.append(improved_result)
                
            except Exception as e:
                print(f"Error in improved NLP processing: {e}")
                continue
        
        return {"improved_nlp_results": improved_results}

# Apply improved NLP processingimproved_nlp = ner_extraction.map_batches(
    AdvancedNLPProcessor(, batch_format="pandas"),
    batch_size=16,
    concurrency=2
)
```

### 7. Ray Data LLM Package Integration (optional)

```python
# Optional: Use Ray Data LLM package for large-scale language model inferencetry:
    from ray.data.llm import LLMPredictor
    
    class LLMTextAnalyzer:
        """Use Ray Data LLM package for improved text analysis."""
        
        def __init__(self, model_name="microsoft/DialoGPT-medium"):
            """Initialize LLM predictor for text analysis."""
            self.model_name = model_name
            
            # Initialize LLM predictor with Ray Data LLM package
            self.llm_predictor = LLMPredictor.from_checkpoint(
                checkpoint=model_name,
                torch_dtype="auto",
                trust_remote_code=True
            )
        
        def __call__(self, batch):
            """Perform LLM-based text analysis."""
            llm_results = []
            
            for item in batch["improved_nlp_results"]:
                try:
                    original_text = item["original_text"]
                    
                    # Create prompts for different analysis tasks
                    analysis_prompts = [
                        f"Analyze the sentiment and key themes in this text: {original_text[:500]}",
                        f"Extract the main topics and entities from: {original_text[:500]}",
                        f"Provide a brief summary of: {original_text[:500]}"
                    ]
                    
                    llm_analyses = []
                    for prompt in analysis_prompts:
                        try:
                            # Use LLM predictor for inference
                            response = self.llm_predictor.predict(prompt)
                            llm_analyses.append({
                                "prompt_type": prompt.split(':')[0],
                                "response": response,
                                "prompt_length": len(prompt)
                            })
                        except Exception as e:
                            llm_analyses.append({
                                "prompt_type": prompt.split(':')[0],
                                "response": f"LLM inference failed: {str(e)}",
                                "error": True
                            })
                    
                    llm_result = {
                        **item,
                        "llm_analyses": llm_analyses,
                        "llm_model_used": self.model_name,
                        "llm_processing_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    llm_results.append(llm_result)
                    
                except Exception as e:
                    print(f"Error in LLM processing: {e}")
                    llm_results.append({
                        **item,
                        "llm_error": str(e)
                    })
            
            return {"llm_results": llm_results}
    
    # Apply LLM analysis (optional - requires Ray Data LLM package)
    llm_analysis = improved_nlp.map_batches(
        LLMTextAnalyzer(, batch_format="pandas"),
        batch_size=8,  # Smaller batch for LLM processing
        num_gpus=1 if ray.cluster_resources().get("GPU", 0) > 0 else 0,
        concurrency=1  # Single concurrency for LLM to avoid resource conflicts
    )
    
    print("LLM analysis completed using Ray Data LLM package")
    
except ImportError:
    print("Ray Data LLM package not available. Skipping LLM analysis.")
    print("To use LLM features, install with: pip install ray[data,llm]")
    llm_analysis = improved_nlp

# Alternative: Simple LLM integration without Ray Data LLM package
class SimpleLLMProcessor:
    """Simple LLM integration using transformers directly."""
    
    def __init__(self):
        from transformers import pipeline
        
        # Initialize text generation pipeline
        self.text_generator = pipeline(
            "text-generation", 
            model="gpt2", 
            device=-1,
            max_length=100
        )
        
        # Initialize summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
    
    def __call__(self, batch):
        """Apply simple LLM processing."""
        simple_llm_results = []
        
        for item in batch["improved_nlp_results"]:
            try:
                text = item["original_text"]
                
                # Text summarization
                summary = ""
                if len(text) > 200:
                    try:
                        summary_result = self.summarizer(text[:1024], max_length=100, min_length=30)
                        summary = summary_result[0]['summary_text']
                    except Exception as e:
                        summary = f"Summarization error: {str(e)}"
                
                # Simple text generation
                generation_prompt = f"Based on this text: {text[:100]}... the key insight is"
                try:
                    generation_result = self.text_generator(generation_prompt, max_length=50, num_return_sequences=1)
                    generated_insight = generation_result[0]['generated_text'][len(generation_prompt):].strip()
                except Exception as e:
                    generated_insight = f"Generation error: {str(e)}"
                
                simple_llm_result = {
                    **item,
                    "text_summary": summary,
                    "generated_insight": generated_insight,
                    "llm_processing_method": "transformers_direct",
                    "llm_timestamp": pd.Timestamp.now().isoformat()
                }
                
                simple_llm_results.append(simple_llm_result)
                
            except Exception as e:
                print(f"Error in simple LLM processing: {e}")
                continue
        
        return {"simple_llm_results": simple_llm_results}

# Apply simple LLM processing as alternativesimple_llm_analysis = improved_nlp.map_batches(
    SimpleLLMProcessor(, batch_format="pandas"),
    batch_size=8,
    concurrency=2
)
```

### 8. Text Similarity and Semantic Search

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextSimilarityAnalyzer:
    """Analyze text similarity and enable semantic search."""
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def __call__(self, batch):
        """Calculate text similarity within batch."""
        similarity_results = []
        
        # Extract embeddings from batch
        embeddings = []
        texts = []
        items = []
        
        for item in batch.get("embedding_results", []):
            if "embeddings" in item and item["embeddings"]:
                embeddings.append(np.array(item["embeddings"]))
                texts.append(item["processed_text"])
                items.append(item)
        
        if len(embeddings) < 2:
            return {"similarity_results": items}
        
        # Calculate pairwise similarities
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find similar text pairs
        for i, item in enumerate(items):
            similar_texts = []
            
            for j, other_item in enumerate(items):
                if i != j and similarity_matrix[i][j] > self.similarity_threshold:
                    similar_texts.append({
                        "similar_text_id": other_item.get("document_id", j),
                        "similarity_score": float(similarity_matrix[i][j]),
                        "similar_text_preview": other_item["processed_text"][:100]
                    })
            
            # Sort by similarity score
            similar_texts.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            result = {
                **item,
                "similar_texts": similar_texts[:5],  # Top 5 similar texts
                "similarity_count": len(similar_texts),
                "has_similar_content": len(similar_texts) > 0
            }
            
            similarity_results.append(result)
        
        return {"similarity_results": similarity_results}

# Apply similarity analysissimilarity_analysis = ner_extraction.map_batches(
    TextSimilarityAnalyzer(, batch_format="pandas"),
    batch_size=20,
    concurrency=2
)
```

### 9. Language Detection and Multilingual Processing

```python
from transformers import pipeline

class MultilingualProcessor:
    """Handle multilingual text processing."""
    
    def __init__(self):
        # Language detection pipeline
        self.language_detector = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=-1
        )
        
        # Multilingual sentiment analysis
        self.multilingual_sentiment = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=-1
        )
    
    def __call__(self, batch):
        """Process multilingual text."""
        multilingual_results = []
        
        for item in batch["similarity_results"]:
            try:
                text = item["processed_text"]
                original_text = item["original_text"]
                
                # Detect language
                try:
                    lang_result = self.language_detector(text[:512])
                    detected_language = lang_result[0]['label']
                    language_confidence = lang_result[0]['score']
                except Exception as e:
                    detected_language = "unknown"
                    language_confidence = 0.0
                
                # Multilingual sentiment analysis
                try:
                    multilingual_sentiment = self.multilingual_sentiment(text[:512])
                    ml_sentiment = multilingual_sentiment[0]['label']
                    ml_sentiment_score = multilingual_sentiment[0]['score']
                except Exception as e:
                    ml_sentiment = "unknown"
                    ml_sentiment_score = 0.0
                
                # Text complexity analysis
                complexity_metrics = {
                    "character_count": len(original_text),
                    "word_count": len(text.split()),
                    "unique_words": len(set(text.split())),
                    "lexical_diversity": len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0,
                    "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if len(text.split()) > 0 else 0
                }
                
                multilingual_result = {
                    **item,
                    "detected_language": detected_language,
                    "language_confidence": language_confidence,
                    "multilingual_sentiment": ml_sentiment,
                    "multilingual_sentiment_score": ml_sentiment_score,
                    "complexity_metrics": complexity_metrics,
                    "is_english": detected_language.lower() in ['en', 'english'],
                    "is_complex_text": complexity_metrics["lexical_diversity"] > 0.6,
                    "multilingual_timestamp": pd.Timestamp.now().isoformat()
                }
                
                multilingual_results.append(multilingual_result)
                
            except Exception as e:
                print(f"Error in multilingual processing: {e}")
                continue
        
        return {"multilingual_results": multilingual_results}

# Apply multilingual processingmultilingual_analysis = similarity_analysis.map_batches(
    MultilingualProcessor(),
    batch_size=16,
    concurrency=2
)
```

### 10. Ray Data LLM Package for Production Inference

```python
# Production-scale LLM inference using Ray Data LLM packagetry:
    from ray.data.llm import LLMPredictor
    
    class ProductionLLMAnalyzer:
        """Production-scale LLM analysis using Ray Data LLM package."""
        
        def __init__(self, model_name="microsoft/DialoGPT-small"):
            """Initialize production LLM analyzer."""
            self.model_name = model_name
            
            # Configure LLM predictor for production use
            self.llm_predictor = LLMPredictor.from_checkpoint(
                checkpoint=model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                max_batch_size=16,
                max_concurrent_requests=4
            )
        
        def __call__(self, batch):
            """Perform production LLM inference."""
            production_results = []
            
            # Prepare prompts for batch inference
            prompts = []
            items = []
            
            for item in batch["multilingual_results"]:
                text = item["original_text"]
                
                # Create structured prompt for analysis
                analysis_prompt = f"""
                Analyze this text and provide insights:
                
                Text: {text[:800]}
                
                Please provide:
                1. Main theme
                2. Key entities
                3. Emotional tone
                4. Business relevance
                """
                
                prompts.append(analysis_prompt)
                items.append(item)
            
            # Batch LLM inference using Ray Data LLM package
            try:
                llm_responses = self.llm_predictor.predict_batch(prompts)
                
                for item, response in zip(items, llm_responses):
                    production_result = {
                        **item,
                        "llm_analysis": response,
                        "llm_model": self.model_name,
                        "llm_method": "ray_data_llm_package",
                        "production_timestamp": pd.Timestamp.now().isoformat()
                    }
                    production_results.append(production_result)
                    
            except Exception as e:
                print(f"LLM batch inference failed: {e}")
                # Fallback to individual processing
                for item in items:
                    production_results.append({
                        **item,
                        "llm_analysis": "LLM processing unavailable",
                        "llm_error": str(e)
                    })
            
            return {"production_llm_results": production_results}
    
    # Apply production LLM analysis
    production_llm = multilingual_analysis.map_batches(
        ProductionLLMAnalyzer(),
        batch_size=8,
        num_gpus=1 if ray.cluster_resources().get("GPU", 0) > 0 else 0,
        concurrency=1
    )
    
    print("Production LLM analysis completed using Ray Data LLM package")
    
except ImportError:
    print("Ray Data LLM package not available.")
    print("To use production LLM features, install with: pip install ray[data] vllm")
    production_llm = multilingual_analysis

# Final results collectionfinal_nlp_results = production_llm.take_all()
print(f"Complete NLP pipeline processed {len(final_nlp_results)} documents")
```

## Advanced Features

### Ray Data's NLP Superpowers

**1. Distributed Model Inference**
```python
# Process 100K documents across cluster with automatic load balancinglarge_dataset.map_batches(
    TransformerModel(), 
    batch_size=64,      # Optimal GPU utilization
    concurrency=8,      # Parallel model instances
    num_gpus=1          # GPU acceleration per instance
)
```

**2. Memory-Efficient Text Processing**
```python
# Handle datasets larger than cluster memorymassive_text_dataset.map_batches(
    TextProcessor(),
    batch_size=1000,    # Process in manageable chunks
    concurrency=16      # Maximize CPU utilization
)
# Ray Data automatically manages memory and spilling```

**3. Fault-Tolerant NLP Pipelines**
```python
# Built-in error recovery for production NLPnlp_pipeline = (dataset
    .map_batches(TextCleaner(,
    num_cpus=0.25
))      # Automatic retry on failures
    .map_batches(SentimentAnalyzer(,
    num_cpus=0.25
)) # Worker failure recovery
    .map_batches(TopicModeler(,
    num_cpus=0.25
))     # Data block replication
)
# No additional fault tolerance code needed```

**4. Model scaling**
```python
# Scale from single GPU to multi-GPU automaticallyif ray.cluster_resources().get("GPU", 0) >= 4:
    # Multi-GPU configuration
    concurrency = 4
    num_gpus = 1
else:
    # CPU fallback
    concurrency = 8
    num_gpus = 0

# Ray Data handles resource allocation automatically```

### Enterprise NLP Use Case Patterns

**Customer Feedback Analysis Pipeline**
- **Scale**: Process 50K+ daily reviews
- **Speed**: 2-hour processing vs 40+ hours manual
- **Accuracy**: Consistent analysis across all content
- **Insights**: Product satisfaction, feature requests, issue identification

**Support Ticket Intelligence**
- **Automation**: 90% tickets auto-classified and routed
- **Response Time**: 15-minute vs 24-hour issue identification
- **Efficiency**: reduction in manual ticket triage
- **Quality**: Consistent urgency and intent classification

**Brand Monitoring at Scale**
- **Coverage**: 100% social media mention analysis
- **Real-time**: Immediate sentiment tracking and alerts
- **Competitive**: Multi-brand comparison and positioning
- **Actionable**: Trend identification and response recommendations

## Production Considerations

### Model Management
- Model versioning and deployment
- A/B testing for model performance
- Model monitoring and retraining

### Scalability
- Horizontal scaling across multiple nodes
- Load balancing for NLP workloads
- Resource optimization for different model types

### Quality Assurance
- Text data validation
- Model performance monitoring
- Output quality checks

## Example Workflows

### 1. Customer Experience Intelligence
**Use Case**: E-commerce company analyzing 50K daily reviews

```python
# Load customer reviews with large datasetsreviews = from_huggingface("amazon_reviews_multi", split="train[:50000]")

# Multi-dimensional sentiment analysissentiment_pipeline = reviews.map_batches(SentimentAnalyzer(,
    num_cpus=0.25
), batch_size=100, concurrency=8)

# Product feature extractionfeature_pipeline = sentiment_pipeline.map_batches(ProductFeatureExtractor(,
    num_cpus=0.25
), batch_size=50)

# Business insights generationinsights = feature_pipeline.groupby('product_category').agg({
    'sentiment_score': 'mean',
    'feature_mentions': 'count',
    'recommendation_score': 'mean'
})

# Results: Product satisfaction by category, feature improvement priorities```

### 2. Brand Monitoring and Competitive Analysis
**Use Case**: Marketing team tracking 25K daily social mentions

```python
# Load social media mentionssocial_data = read_text("s3://social-media-feeds/mentions/")

# Brand sentiment trackingbrand_analysis = social_data.map_batches(BrandSentimentAnalyzer(,
    num_cpus=0.25
), batch_size=200)

# Competitive comparisoncompetitor_analysis = brand_analysis.map_batches(CompetitorMentionTracker(,
    num_cpus=0.25
), batch_size=100)

# Trend identificationtrending_topics = competitor_analysis.groupby('mention_type').agg({
    'sentiment_score': 'mean',
    'engagement_rate': 'mean',
    'virality_score': 'max'
})

# Results: Real-time brand health, competitive positioning insights```

### 3. Support Ticket Intelligence
**Use Case**: Customer service team processing 15K daily tickets

```python
# Load support tickets# Support tickets - JSON format (typical for customer support systems)tickets = ray.data.read_json("s3://ray-benchmark-data/support/tickets/*.json",
    num_cpus=0.05
)

# Urgency classification and routingclassified_tickets = tickets.map_batches(TicketClassifier(,
    num_cpus=0.25
), batch_size=150)

# Issue categorization and solution matchingcategorized_tickets = classified_tickets.map_batches(IssueCategorizer(,
    num_cpus=0.25
), batch_size=100)

# Knowledge base enhancementknowledge_updates = categorized_tickets.map_batches(KnowledgeExtractor(,
    num_cpus=0.25
), batch_size=75)

# Results: Automated ticket routing, solution recommendations, knowledge base updates```

### 4. Content Moderation at Scale
**Use Case**: Social platform moderating 100K daily posts

```python
# Load user-generated contentuser_content = read_json("s3://platform-content/posts/",
    num_cpus=0.05
)

# Multi-layer content analysissafety_analysis = user_content.map_batches(ContentSafetyAnalyzer(,
    num_cpus=0.25
), batch_size=500)

# Toxicity and harmful content detectiontoxicity_analysis = safety_analysis.map_batches(ToxicityDetector(,
    num_cpus=0.25
), batch_size=300)

# Automated moderation decisionsmoderation_actions = toxicity_analysis.map_batches(ModerationDecisionEngine(,
    num_cpus=0.25
), batch_size=200)

# Results: Automated content moderation, safety scoring, action recommendations```

### 5. Document Intelligence and Compliance
**Use Case**: Legal firm processing 10K daily documents

```python
# Load legal documentslegal_docs = read_text("s3://legal-documents/filings/")

# Contract analysis and clause extractioncontract_analysis = legal_docs.map_batches(ContractAnalyzer(,
    num_cpus=0.25
), batch_size=50)

# Compliance checking and risk assessmentcompliance_check = contract_analysis.map_batches(ComplianceValidator(,
    num_cpus=0.25
), batch_size=25)

# Key information extraction for case managementcase_intelligence = compliance_check.map_batches(CaseIntelligenceExtractor(,
    num_cpus=0.25
), batch_size=30)

# Results: Contract risk assessment, compliance validation, case insights```

### 6. Multilingual Customer Support
**Use Case**: Global company handling 30K daily multilingual inquiries

```python
# Load multilingual customer inquiriesinquiries = read_text("s3://global-support/inquiries/")

# Language detection and routinglanguage_analysis = inquiries.map_batches(LanguageDetector(,
    num_cpus=0.25
), batch_size=300)

# Multilingual sentiment and intent analysisintent_analysis = language_analysis.map_batches(MultilingualIntentAnalyzer(,
    num_cpus=0.25
), batch_size=150)

# Automated response generationresponse_generation = intent_analysis.map_batches(ResponseGenerator(,
    num_cpus=0.25
), batch_size=100)

# Results: Automated multilingual support, intent classification, response suggestions```

## Performance Analysis

### NLP Pipeline Processing Flow

```
Text Data Processing Pipeline:

  Raw Text Data   (IMDB, Amazon, News)
  1M+ documents  

         
         
        
 Text Cleaning        BERT Embeddings      Sentiment       
 & Preprocessing  Generation       Analysis        
                      (384-dim)            (Pos/Neg/Neu)  
        
                                                       
                 
          Named Entity         Topic Modeling      
          Recognition      & Clustering    
          (Persons/Orgs)       (LDA/K-means)   
             
```

### Performance Measurement Framework

| Analysis Type | Benchmark Method | Visualization Output |
|--------------|------------------|---------------------|
| **Text Processing** | Throughput measurement | Processing speed charts |
| **Model Inference** | GPU utilization tracking | Resource usage graphs |
| **Sentiment Analysis** | Accuracy vs speed | Performance trade-offs |
| **Topic Modeling** | Convergence analysis | Topic quality metrics |

### Expected Output Visualizations

The demo generates comprehensive analysis charts:

| Chart Type | File Output | Content Description |
|-----------|-------------|-------------------|
| **Sentiment Distribution** | `sentiment_analysis.html` | Positive/Negative/Neutral breakdown |
| **Topic Modeling Results** | `topic_visualization.html` | Word clouds and topic clusters |
| **Performance Metrics** | `nlp_performance.html` | Throughput and resource usage |
| **Entity Analysis** | `entity_extraction.html` | Named entity frequency charts |

### Sample Output Structure

```
NLP Analysis Results:
 Text Statistics
    Document count: [Actual count]
    Average length: [Measured]
    Language distribution: [Detected]
 Sentiment Analysis
    Positive: [%]
    Negative: [%]
    Neutral: [%]
 Topic Modeling
    Topics discovered: [Number]
    Top keywords per topic
    Document-topic assignments
 Named Entities
     Persons: [Count]
     Organizations: [Count]
     Locations: [Count]
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution | Prevention |
|-------|----------|----------|------------|
| **Model Download Failures** | `OSError: Can't load model` | Check internet connection, try different model | Use local model cache, verify model names |
| **GPU Memory Issues** | `CUDA out of memory` | Reduce batch size to 8-16, use CPU fallback | Monitor GPU memory, start with small batches |
| **Text Encoding Errors** | `UnicodeDecodeError` | Add encoding handling, clean text | Validate text encoding, use UTF-8 |
| **Slow Processing** | Long processing times | Use GPU acceleration, optimize batch size | Profile operations, check resource utilization |
| **Model Compatibility** | Version conflicts | Update transformers library | Pin dependency versions in requirements.txt |

### Performance Optimization for Different Text Sizes

```python
# Adaptive batch sizing based on text characteristicsdef get_optimal_batch_size(sample_texts):
    """Calculate optimal batch size based on text characteristics."""
    avg_length = sum(len(text) for text in sample_texts) / len(sample_texts)
    
    if avg_length > 1000:
        return 4   # Long texts - small batches
    elif avg_length > 500:
        return 8   # Medium texts - medium batches
    else:
        return 16  # Short texts - larger batches

# Memory-efficient text processingdef process_with_memory_management(dataset, processor_class):
    """Process text with automatic memory management."""
    import torch
    
    # Start with conservative batch size
    batch_size = 8
    
    try:
        result = dataset.map_batches(
            processor_class(),
            batch_size=batch_size,
            num_gpus=1 if torch.cuda.is_available() else 0
        )
        return result
        
    except torch.cuda.OutOfMemoryError:
        print("GPU memory error, reducing batch size and retrying...")
        torch.cuda.empty_cache()
        
        # Retry with smaller batch
        return dataset.map_batches(
            processor_class(),
            batch_size=batch_size // 2,
            num_gpus=0  # Use CPU fallback
        )
```

### Text Data Quality Validation

```python
# Validate text data quality before processingdef validate_text_quality(batch):
    """Validate text data quality and clean problematic entries."""
    clean_texts = []
    
    for item in batch:
        text = item.get('text', '')
        
        # Quality checks
        issues = []
        
        if not text or len(text.strip()) == 0:
            issues.append("empty_text")
        
        if len(text) > 10000:
            issues.append("text_too_long")
            text = text[:10000]  # Truncate
        
        if not text.isascii():
            try:
                text = text.encode('ascii', 'ignore').decode('ascii')
                issues.append("non_ascii_characters")
            except:
                issues.append("encoding_error")
                continue
        
        clean_item = {
            **item,
            'text': text,
            'quality_issues': issues,
            'is_clean': len(issues) == 0
        }
        
        clean_texts.append(clean_item)
    
    return clean_texts

# Apply validation before processingvalidated_texts = text_dataset.map_batches(validate_text_quality, num_cpus=0.5, batch_size=100)
clean_texts = validated_texts.filter(lambda x: x['is_clean'],
    num_cpus=0.1
)
```

### Debug Mode and Monitoring

```python
import logging
import torch
from transformers import logging as transformers_logging

# Enable comprehensive debugginglogging.basicConfig(level=logging.DEBUG)
transformers_logging.set_verbosity_info()

# Simple timing for performance trackingdef track_processing_time():
    """Track processing time for performance analysis."""
    return time.time()

print("Use Ray Dashboard for detailed resource monitoring and cluster metrics")

# Enable progress trackingclass ProgressTracker:
    def __init__(self, total_items):
        self.total_items = total_items
        self.processed_items = 0
    
    def update(self, batch_size):
        self.processed_items += batch_size
        progress = (self.processed_items / self.total_items) * 100
        print(f"Progress: {progress:.1f}% ({self.processed_items:,}/{self.total_items:,})")
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or increase cluster memory allocation
2. **Slow Processing**: Optimize batch sizes and enable parallel text processing
3. **Model Loading Failures**: Ensure NLP models are accessible to all workers
4. **Text Encoding Issues**: Handle Unicode and special characters properly

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ray.data.context import DataContext
ctx = DataContext.get_current()
ctx.enable_progress_bars = True
```

## Performance Considerations

- Use Ray Dashboard to monitor throughput and resource utilization.
- Tune batch sizes and concurrency based on your actual cluster resources and dataset size.

## Key Takeaways

- **Ray Data democratizes large-scale NLP**: Enterprise text processing without complex infrastructure
- **Distributed processing essential for modern text volumes**: Social media and document analysis require parallel processing
- **Preprocessing optimization provides major gains**: Proper text cleaning and tokenization improve downstream performance
- **Production NLP requires monitoring**: Text data quality and model performance need continuous validation

## Action Items

### Immediate Goals (Next 2 weeks)
1. **Implement text analytics pipeline** for your specific document processing needs
2. **Add sentiment analysis** to understand text sentiment with large datasets
3. **Set up text preprocessing** with tokenization and cleaning
4. **Create text quality monitoring** to ensure processing accuracy

### Long-term Goals (Next 3 months)
1. **Deploy production NLP systems** with real-time text processing
2. **Implement improved NLP features** like entity recognition and topic modeling
3. **Build text search and recommendation** systems using embeddings
4. **Create multilingual support** for global text analysis

## Resources

- [Ray Data Documentation](https://docs.ray.io/en/latest/data/index.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [spaCy Documentation](https://spacy.io/usage)
- [NLTK Documentation](https://www.nltk.org/)

## Cleanup and Resource Management

```python
# Clean up Ray resources
ray.shutdown()
print("Ray cluster shutdown complete")
```

---

*This template provides a foundation for enterprise-scale text analytics with Ray Data. Start with basic sentiment analysis and systematically add improved NLP capabilities based on your specific requirements.*
