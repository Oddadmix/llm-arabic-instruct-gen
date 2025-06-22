"""
Text chunking module for splitting documents into manageable chunks.
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking for document processing, including Arabic support."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        logger.info(f"Starting text chunking. Input text length: {len(text)} characters")
        
        # Clean and normalize text
        logger.info("Cleaning and normalizing text...")
        text = self._clean_text(text)
        logger.info(f"Text cleaned. Length after cleaning: {len(text)} characters")
        
        chunks = []
        start = 0
        chunk_count = 0
        
        logger.info("Creating chunks with overlap...")
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(end + 50, len(text))
                
                # Find the last sentence ending in this range
                sentence_end = self._find_sentence_boundary(
                    text[search_start:search_end], search_start
                )
                
                if sentence_end and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end
                    logger.debug(f"Chunk {chunk_count}: breaking at sentence boundary")
                else:
                    logger.debug(f"Chunk {chunk_count}: breaking at character limit")
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
                logger.debug(f"Created chunk {chunk_count}: {len(chunk)} characters")
            
            # Move start position for next chunk with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        logger.info(f"Chunking completed. Created {len(chunks)} chunks")
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
            logger.info(f"Chunk size range: {min(len(chunk) for chunk in chunks)} - {max(len(chunk) for chunk in chunks)} characters")
        
        return chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[str]:
        """Split text into chunks based on sentence boundaries, supporting Arabic."""
        if not text.strip():
            logger.warning("Empty text provided for sentence-based chunking")
            return []
        
        logger.info(f"Starting sentence-based chunking. Input text length: {len(text)} characters")
        logger.info(f"Sentences per chunk: {sentences_per_chunk}")
        
        # Clean and normalize text
        logger.info("Cleaning and normalizing text...")
        text = self._clean_text(text)
        logger.info(f"Text cleaned. Length after cleaning: {len(text)} characters")
        
        # Split into sentences
        logger.info("Splitting text into sentences...")
        sentences = self._split_sentences(text)
        logger.info(f"Found {len(sentences)} sentences")
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk = " ".join(chunk_sentences).strip()
            if chunk:
                chunks.append(chunk)
                logger.debug(f"Created sentence-based chunk {len(chunks)}: {len(chunk)} characters")
        
        logger.info(f"Sentence-based chunking completed. Created {len(chunks)} chunks")
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        logger.debug("Cleaning text...")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep Arabic and English letters, numbers, and common punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\؟\؛\:،\-\(\)\[\]\{\}ء-ي]', '', text)
        
        cleaned_length = len(text.strip())
        logger.debug(f"Text cleaned. Length: {cleaned_length} characters")
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, supporting Arabic and English punctuation."""
        logger.debug("Splitting text into sentences...")
        
        # Split on English and Arabic sentence-ending punctuation
        # . ! ? ؟ ؛
        sentences = re.split(r'[.!?؟؛]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.debug(f"Split into {len(sentences)} sentences")
        return sentences
    
    def _find_sentence_boundary(self, text_segment: str, offset: int) -> Optional[int]:
        """Find the last sentence boundary in a text segment, supporting Arabic."""
        # Look for sentence endings (., !, ?, ؟, ؛)
        for i in range(len(text_segment) - 1, -1, -1):
            if text_segment[i] in '.!?؟؛':
                return offset + i + 1
        return None 