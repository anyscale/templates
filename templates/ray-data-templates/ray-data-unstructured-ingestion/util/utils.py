"""Utility functions for unstructured data ingestion."""

import matplotlib.pyplot as plt
import numpy as np


def visualize_document_processing():
    """Create concise document processing analytics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Document types processed
    doc_types = ['Financial', 'Legal', 'Regulatory', 'Client', 'Research', 'General']
    counts = [145, 89, 67, 123, 98, 201]
    colors = ['green', 'blue', 'purple', 'orange', 'red', 'gray']
    
    bars = axes[0].bar(doc_types, counts, color=colors, alpha=0.7)
    axes[0].set_title('Documents by Type', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    # 2. Quality distribution
    quality_levels = ['High', 'Medium', 'Low']
    quality_counts = [482, 168, 73]
    colors_qual = ['darkgreen', 'orange', 'red']
    
    wedges, texts, autotexts = axes[1].pie(quality_counts, labels=quality_levels,
                                           colors=colors_qual, autopct='%1.1f%%',
                                           startangle=90)
    axes[1].set_title('Document Quality Distribution', fontweight='bold')
    
    # 3. Processing pipeline stages
    stages = ['Discovered', 'Extracted', 'Chunked', 'Analyzed', 'Warehouse\nReady']
    stage_counts = [723, 695, 682, 671, 643]
    
    axes[2].plot(stages, stage_counts, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[2].fill_between(range(len(stages)), stage_counts, alpha=0.3)
    axes[2].set_title('Pipeline Stage Funnel', fontweight='bold')
    axes[2].set_ylabel('Document Count')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    for i, (stage, cnt) in enumerate(zip(stages, stage_counts)):
        axes[2].text(i, cnt + 10, str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('document_processing.png', dpi=150, bbox_inches='tight')
    print("Document processing visualization saved")
    
    return fig

from typing import Dict, Any
import hashlib
from datetime import datetime


def extract_document_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from document record."""
    
    file_path = record.get('path', '')
    bytes_data = record.get('bytes', b'')
    
    # Extract file information
    file_name = file_path.split('/')[-1] if file_path else 'unknown'
    file_extension = file_name.split('.')[-1] if '.' in file_name else 'unknown'
    file_size_mb = len(bytes_data) / (1024 * 1024) if bytes_data else 0
    
    # Generate document ID
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    # Classify document type
    doc_types = {
        'pdf': 'document',
        'docx': 'document',
        'doc': 'document',
        'txt': 'text',
        'md': 'text',
        'json': 'structured',
        'csv': 'structured',
        'xlsx': 'spreadsheet'
    }
    
    doc_type = doc_types.get(file_extension.lower(), 'unknown')
    
    metadata = {
        'document_id': doc_id,
        'file_name': file_name,
        'file_path': file_path,
        'file_extension': file_extension,
        'file_size_mb': round(file_size_mb, 2),
        'document_type': doc_type,
        'processing_timestamp': datetime.now().isoformat(),
        'bytes': bytes_data  # Preserve original data
    }
    
    return metadata


def create_text_chunks_for_analytics(record: Dict[str, Any]) -> list:
    """Create text chunks suitable for analytics processing."""
    
    text_content = record.get('text_content', '')
    
    if not text_content or len(text_content) < 100:
        return []
    
    # Chunk by paragraphs or fixed size
    chunk_size = 512
    chunks = []
    
    # Simple chunking strategy
    for i in range(0, len(text_content), chunk_size):
        chunk_text = text_content[i:i + chunk_size]
        
        if len(chunk_text) < 50:  # Skip very small chunks
            continue
        
        chunk = {
            'document_id': record.get('document_id', 'unknown'),
            'chunk_id': f"{record.get('document_id', 'unknown')}_{i // chunk_size}",
            'chunk_text': chunk_text,
            'chunk_position': i // chunk_size,
            'chunk_size': len(chunk_text),
            'source_file': record.get('file_name', 'unknown')
        }
        
        chunks.append(chunk)
    
    return chunks

