"""Utility functions for unstructured data ingestion."""

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

