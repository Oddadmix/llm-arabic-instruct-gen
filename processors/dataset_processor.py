"""
Dataset processing module for handling Hugging Face datasets with pre-chunked text data.
"""

from typing import List, Optional, Dict, Any
import logging
from datasets import load_dataset, Dataset
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Handles Hugging Face datasets with pre-chunked text data."""
    
    def __init__(self, text_column: str = "text", max_samples: Optional[int] = None, start_index: int = 0):
        self.text_column = text_column
        self.max_samples = max_samples
        self.start_index = start_index
        
        logger.info(f"DatasetProcessor initialized with:")
        logger.info(f"  - Text column: {text_column}")
        logger.info(f"  - Max samples: {max_samples if max_samples else 'unlimited'}")
        logger.info(f"  - Start index: {start_index}")
    
    def load_dataset(self, dataset_name: str, split: str = "train", 
                    config_name: Optional[str] = None) -> List[str]:
        """Load a Hugging Face dataset and extract text chunks from the specified column."""
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        logger.info(f"  - Split: {split}")
        logger.info(f"  - Config: {config_name if config_name else 'default'}")
        logger.info(f"  - Text column: {self.text_column}")
        logger.info(f"  - Start index: {self.start_index}")
        
        try:
            # Load the dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            logger.info(f"Dataset loaded successfully")
            logger.info(f"  - Dataset type: {type(dataset)}")
            logger.info(f"  - Available columns: {list(dataset.column_names)}")
            logger.info(f"  - Total samples: {len(dataset)}")
            
            # Validate text column exists
            if self.text_column not in dataset.column_names:
                available_columns = list(dataset.column_names)
                raise ValueError(f"Text column '{self.text_column}' not found in dataset. "
                               f"Available columns: {available_columns}")
            
            # Validate start index
            if self.start_index >= len(dataset):
                raise ValueError(f"Start index {self.start_index} is out of range. "
                               f"Dataset has {len(dataset)} samples.")
            
            # Extract text chunks
            text_chunks = self._extract_text_chunks(dataset)
            
            logger.info(f"Text extraction completed:")
            logger.info(f"  - Total chunks extracted: {len(text_chunks)}")
            logger.info(f"  - Average chunk length: {sum(len(chunk) for chunk in text_chunks) / len(text_chunks):.0f} characters")
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def load_dataset_from_file(self, file_path: str, file_type: str = "auto") -> List[str]:
        """Load a dataset from a local file (CSV, JSON, Parquet, etc.)."""
        logger.info(f"Loading dataset from file: {file_path}")
        logger.info(f"  - File type: {file_type}")
        logger.info(f"  - Text column: {self.text_column}")
        logger.info(f"  - Start index: {self.start_index}")
        
        try:
            # Load dataset from file
            if file_type == "auto":
                # Auto-detect file type from extension
                file_extension = file_path.lower().split('.')[-1]
                if file_extension in ['csv']:
                    file_type = 'csv'
                elif file_extension in ['json', 'jsonl']:
                    file_type = 'json'
                elif file_extension in ['parquet']:
                    file_type = 'parquet'
                else:
                    file_type = 'csv'  # Default to CSV
            
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == 'csv':
                dataset = load_dataset('csv', data_files=file_path, split='train')
            elif file_type == 'json':
                dataset = load_dataset('json', data_files=file_path, split='train')
            elif file_type == 'jsonl':
                dataset = load_dataset('json', data_files=file_path, split='train')
            elif file_type == 'parquet':
                dataset = load_dataset('parquet', data_files=file_path, split='train')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Dataset loaded successfully from file")
            logger.info(f"  - Dataset type: {type(dataset)}")
            logger.info(f"  - Available columns: {list(dataset.column_names)}")
            logger.info(f"  - Total samples: {len(dataset)}")
            
            # Validate text column exists
            if self.text_column not in dataset.column_names:
                available_columns = list(dataset.column_names)
                raise ValueError(f"Text column '{self.text_column}' not found in dataset. "
                               f"Available columns: {available_columns}")
            
            # Validate start index
            if self.start_index >= len(dataset):
                raise ValueError(f"Start index {self.start_index} is out of range. "
                               f"Dataset has {len(dataset)} samples.")
            
            # Extract text chunks
            text_chunks = self._extract_text_chunks(dataset)
            
            logger.info(f"Text extraction completed:")
            logger.info(f"  - Total chunks extracted: {len(text_chunks)}")
            logger.info(f"  - Average chunk length: {sum(len(chunk) for chunk in text_chunks) / len(text_chunks):.0f} characters")
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error loading dataset from file {file_path}: {str(e)}")
            raise
    
    def _extract_text_chunks(self, dataset: Dataset) -> List[tuple]:
        """Extract text chunks from the dataset, returning (real_index, chunk) tuples."""
        logger.info("Extracting text chunks from dataset...")
        
        text_chunks = []
        total_samples = len(dataset)
        start_idx = self.start_index
        end_idx = total_samples
        
        if self.max_samples:
            end_idx = min(total_samples, start_idx + self.max_samples)
        
        samples_to_process = end_idx - start_idx
        
        logger.info(f"Processing {samples_to_process} samples from index {start_idx} to {end_idx-1} out of {total_samples}")
        
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            text = sample.get(self.text_column, "")
            if text and isinstance(text, str) and text.strip():
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    text_chunks.append((i, cleaned_text))  # i is the real dataset index
                    logger.debug(f"Sample {i}: extracted {len(cleaned_text)} characters")
                else:
                    logger.debug(f"Sample {i}: empty after cleaning")
            else:
                logger.debug(f"Sample {i}: no valid text found")
            if (i - start_idx + 1) % 1000 == 0:
                logger.info(f"Progress: {i - start_idx + 1}/{samples_to_process} samples processed, {len(text_chunks)} chunks extracted")
        logger.info(f"Text chunk extraction completed: {len(text_chunks)} chunks extracted")
        return text_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text chunk."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove very short texts (likely noise)
        if len(text.strip()) < 10:
            return ""
        
        # Add section header for dataset chunks
        section_header = f"\n\n=== DATASET CHUNK ===\n\n"
        section_footer = f"\n\n=== END CHUNK ===\n\n"
        
        return section_header + text.strip() + section_footer
    
    def get_dataset_info(self, dataset_name: str, split: str = "train", 
                        config_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a dataset without loading all data."""
        logger.info(f"Getting dataset info for: {dataset_name}")
        
        try:
            # Load dataset info
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            info = {
                "name": dataset_name,
                "split": split,
                "config": config_name,
                "total_samples": len(dataset),
                "columns": list(dataset.column_names),
                "features": dataset.features,
                "text_column_exists": self.text_column in dataset.column_names
            }
            
            # Sample a few examples to show structure
            if len(dataset) > 0:
                sample = dataset[0]
                info["sample_data"] = {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) 
                                     for k, v in sample.items()}
            
            logger.info(f"Dataset info retrieved successfully")
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_name}: {str(e)}")
            raise 