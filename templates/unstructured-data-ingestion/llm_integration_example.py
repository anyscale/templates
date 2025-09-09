#!/usr/bin/env python3
"""
Example integration with Ray Data LLM package for production use.

This script demonstrates how to replace the simulated LLM processing
in the main demo with actual Ray Data LLM package calls.
"""

import ray
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Example 1: Basic LLM Inference with Ray Data LLM Package
# =============================================================================

def example_basic_llm_inference():
    """
    Basic example of using Ray Data LLM package for inference.
    """
    try:
        # Import the LLM package (when available)
        from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
        
        # Example dataset with text chunks
        text_chunks = [
            "This is the first document chunk about machine learning.",
            "This is the second chunk about Ray Data processing.",
            "This is the third chunk about distributed computing."
        ]
        
        # Create Ray dataset
        ds = ray.data.from_items(text_chunks)
        
        # Configure vLLM engine
        config = vLLMEngineProcessorConfig(
            model_source="meta-llama/Llama-2-7b-chat-hf",
            engine_kwargs={
                "max_model_len": 16384,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            concurrency=1,
            batch_size=2,
        )
        
        # Build processor
        processor = build_llm_processor(
            config,
            preprocess=lambda row: dict(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": row["item"]}
                ],
                sampling_params=dict(
                    temperature=0.7,
                    max_tokens=100,
                )
            ),
            postprocess=lambda row: dict(
                answer=row["generated_text"],
                **row
            ),
        )
        
        # Apply LLM inference
        processed_ds = processor(ds)
        
        logger.info("Basic LLM inference completed successfully")
        return processed_ds
        
    except ImportError:
        logger.warning("Ray Data LLM package not available. This is a placeholder example.")
        return None

# =============================================================================
# Example 2: Custom LLM Processing Function
# =============================================================================

def custom_llm_processor(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Custom LLM processing function that can be used with Ray Data.
    
    Args:
        batch: Batch of text chunks to process
        
    Returns:
        Processed batch with LLM-generated content
    """
    try:
        from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
        
        # Configure vLLM engine
        config = vLLMEngineProcessorConfig(
            model_source="meta-llama/Llama-2-7b-chat-hf",
            engine_kwargs={
                "max_model_len": 16384,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            concurrency=1,
            batch_size=len(batch["text"]),
        )
        
        # Build processor for batch processing
        processor = build_llm_processor(
            config,
            preprocess=lambda row: dict(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes text."},
                    {"role": "user", "content": row["text"]}
                ],
                sampling_params=dict(
                    temperature=0.5,
                    max_tokens=150,
                )
            ),
            postprocess=lambda row: row
        )
        
        # Create dataset from batch
        batch_ds = ray.data.from_items([
            {"text": text} for text in batch["text"]
        ])
        
        # Process with LLM
        processed_ds = processor(batch_ds)
        processed_results = processed_ds.take_all()
        
        # Extract results
        summaries = [result["generated_text"] for result in processed_results]
        entities = [extract_entities_from_text(result["generated_text"]) for result in processed_results]
        sentiments = [analyze_sentiment_from_text(result["generated_text"]) for result in processed_results]
        
        return {
            "original_text": batch["text"],
            "summary": summaries,
            "entities": entities,
            "sentiment": sentiments
        }
        
    except ImportError:
        logger.warning("Ray Data LLM package not available. Using placeholder processing.")
        # Fallback to placeholder processing
        return {
            "original_text": batch["text"],
            "summary": [f"Summary of: {text[:50]}..." for text in batch["text"]],
            "entities": [["placeholder", "entities"] for _ in batch["text"]],
            "sentiment": ["neutral" for _ in batch["text"]]
        }

def extract_entities_from_text(text: str) -> List[str]:
    """Extract entities from text (simplified)."""
    # Simple entity extraction - in production, use more sophisticated parsing
    words = text.split()
    return [word.strip(".,") for word in words if len(word) > 3][:5]

def analyze_sentiment_from_text(text: str) -> str:
    """Analyze sentiment from text (simplified)."""
    # Simple sentiment analysis - in production, use proper sentiment analysis
    positive_words = ["good", "great", "excellent", "positive", "happy"]
    negative_words = ["bad", "terrible", "awful", "negative", "sad"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# =============================================================================
# Example 3: Integration with Main Pipeline
# =============================================================================

def integrate_with_main_pipeline(chunked_dataset):
    """
    Example of how to integrate LLM processing with the main pipeline.
    
    Args:
        chunked_dataset: Ray dataset with chunked text
        
    Returns:
        Processed dataset with LLM-generated content
    """
    logger.info("Integrating LLM processing with main pipeline...")
    
    # Option 1: Use built-in LLM inference with vLLM
    try:
        from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
        
        # Configure vLLM engine
        config = vLLMEngineProcessorConfig(
            model_source="meta-llama/Llama-2-7b-chat-hf",
            engine_kwargs={
                "max_model_len": 16384,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            concurrency=1,
            batch_size=32,
        )
        
        # Build processor
        processor = build_llm_processor(
            config,
            preprocess=lambda row: dict(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes document chunks."},
                    {"role": "user", "content": f"Analyze this text: {row['text']}"}
                ],
                sampling_params=dict(
                    temperature=0.3,
                    max_tokens=200,
                )
            ),
            postprocess=lambda row: dict(
                original_text=row.get("text", ""),
                summary=row.get("generated_text", ""),
                doc_id=row.get("doc_id", ""),
                chunk_id=row.get("chunk_id", ""),
                # ... other fields
            )
        )
        
        # Process with LLM
        llm_processed_ds = processor(chunked_dataset)
        
        logger.info("Using built-in LLM inference with vLLM")
        return llm_processed_ds
        
    except ImportError:
        logger.info("Using custom LLM processing function")
        
        # Option 2: Use custom processing function
        llm_processed_ds = chunked_dataset.map_batches(
            custom_llm_processor,
            batch_size=32,
            concurrency=2,
            num_cpus=1
        )
        
        return llm_processed_ds

# =============================================================================
# Example 4: Advanced LLM Configuration
# =============================================================================

def advanced_llm_configuration():
    """
    Example of advanced LLM configuration options.
    """
    config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_tokens": 300,
        "temperature": 0.1,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stop": ["\n\n", "END"],
        "batch_size": 16,
        "num_gpus": 1,
        "max_concurrent_requests": 4
    }
    
    logger.info(f"Advanced LLM configuration: {config}")
    return config

# =============================================================================
# Example 5: Error Handling and Fallbacks
# =============================================================================

def robust_llm_processing(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Robust LLM processing with error handling and fallbacks.
    
    Args:
        batch: Batch of text chunks
        
    Returns:
        Processed batch with fallback handling
    """
    try:
        from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
        
        # Configure vLLM engine
        config = vLLMEngineProcessorConfig(
            model_source="meta-llama/Llama-2-7b-chat-hf",
            engine_kwargs={
                "max_model_len": 16384,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 2048,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            concurrency=1,
            batch_size=len(batch["text"]),
        )
        
        # Build processor
        processor = build_llm_processor(
            config,
            preprocess=lambda row: dict(
                messages=[
                    {"role": "system", "content": "You are an AI assistant that processes text."},
                    {"role": "user", "content": f"Process this text: {row['text']}"}
                ],
                sampling_params=dict(
                    temperature=0.5,
                    max_tokens=100,
                )
            ),
            postprocess=lambda row: row
        )
        
        # Create dataset from batch
        batch_ds = ray.data.from_items([
            {"text": text} for text in batch["text"]
        ])
        
        # Process with LLM
        processed_ds = processor(batch_ds)
        processed_results = processed_ds.take_all()
        
        # Extract results with error handling
        results = []
        for i, result in enumerate(processed_results):
            try:
                results.append(result["generated_text"])
            except Exception as e:
                logger.warning(f"LLM processing failed for text {i}: {e}")
                # Fallback to simple processing
                results.append(f"Fallback processing: {batch['text'][i][:50]}...")
        
        return {"processed_text": results}
        
    except Exception as e:
        logger.error(f"LLM processing failed completely: {e}")
        # Complete fallback
        return {
            "processed_text": [f"Complete fallback: {text[:50]}..." for text in batch["text"]]
        }

# =============================================================================
# Main execution example
# =============================================================================

if __name__ == "__main__":
    # Initialize Ray
    ray.init()
    
    logger.info("Running LLM integration examples...")
    
    # Run examples
    example_basic_llm_inference()
    
    # Create sample dataset
    sample_texts = [
        "Ray Data is a powerful library for distributed data processing.",
        "Machine learning models can be trained at scale using Ray.",
        "Distributed computing enables processing of large datasets."
    ]
    
    sample_ds = ray.data.from_items(sample_texts)
    
    # Test custom processing
    processed = custom_llm_processor({"text": sample_texts})
    logger.info(f"Custom processing result: {processed}")
    
    # Test robust processing
    robust_result = robust_llm_processing({"text": sample_texts})
    logger.info(f"Robust processing result: {robust_result}")
    
    # Show advanced configuration
    advanced_config = advanced_llm_configuration()
    
    logger.info("LLM integration examples completed")
    
    # Shutdown Ray
    ray.shutdown()
