"""
Embeddings generation utility using transformers library.
"""

import numpy as np
from typing import List, Optional, Union
import logging
import torch
import gc
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text chunks using transformers models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct", 
                 batch_size: int = 32, device: str = "auto",
                 offload_model: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._get_device(device)
        self.offload_model = offload_model
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        logger.info(f"EmbeddingGenerator initialized with:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Offload model: {offload_model}")
        
        # Only load model if not offloading
        if not self.offload_model:
            self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Auto-detected MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("Auto-detected CPU device")
        else:
            device = torch.device(device)
            logger.info(f"Using specified device: {device}")
        return device
    
    def _load_model(self):
        """Load the embedding model and tokenizer."""
        if self.model_loaded:
            logger.debug("Embedding model already loaded")
            return
            
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            if "sentence-transformers" in self.model_name:
                # Use sentence-transformers for better performance
                logger.info("Loading sentence-transformers model...")
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                logger.info(f"Loaded sentence-transformers model: {self.model_name}")
                logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
            else:
                # Use transformers for custom models
                logger.info("Loading transformers model...")
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info("Tokenizer loaded successfully")
                
                logger.info("Loading model...")
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info("Model loaded successfully")
                
                logger.info(f"Moving model to device: {self.device}")
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model moved to device and set to evaluation mode")
                logger.info(f"Model dimension: {self.model.config.hidden_size}")
            
            self.model_loaded = True
            logger.info("Embedding model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _unload_model(self):
        """Unload the embedding model to free memory."""
        if not self.model_loaded or not self.offload_model:
            return
            
        logger.info("Unloading embedding model to free memory...")
        
        if self.model is not None:
            # Move model to CPU first to free GPU memory
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            
            # Delete model
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self.model_loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
        logger.info("Embedding model unloaded successfully")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([])
        
        # Load model if needed
        if self.offload_model and not self.model_loaded:
            self._load_model()
        
        logger.info(f"Starting embedding generation for {len(texts)} texts")
        logger.info(f"Average text length: {sum(len(text) for text in texts) / len(texts):.0f} characters")
        
        try:
            if isinstance(self.model, SentenceTransformer):
                # Use sentence-transformers
                logger.info("Using sentence-transformers for embedding generation...")
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    device=str(self.device)
                )
                logger.info("Sentence-transformers embedding generation completed")
            else:
                # Use transformers with custom model
                logger.info("Using transformers for embedding generation...")
                embeddings = self._generate_with_transformers(texts)
                logger.info("Transformers embedding generation completed")
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            # Unload model after generation if offloading is enabled
            if self.offload_model:
                self._unload_model()
    
    def _generate_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformers library."""
        logger.info("Generating embeddings with transformers...")
        embeddings = []
        
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches of size {self.batch_size}")
        
        for batch_idx in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            current_batch = batch_idx // self.batch_size + 1
            
            logger.debug(f"Processing batch {current_batch}/{total_batches} ({len(batch_texts)} texts)")
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                batch_embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings.append(batch_embeddings.cpu().numpy())
            
            logger.debug(f"Batch {current_batch} completed: {batch_embeddings.shape}")
        
        logger.info("All batches processed successfully")
        return np.vstack(embeddings)
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def similarity_search(self, query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> tuple:
        """Find most similar embeddings to query."""
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for similarity search")
            return [], []
        
        logger.debug(f"Performing similarity search with top_k={top_k}")
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        logger.debug(f"Similarity search completed. Top score: {top_scores[0]:.3f}")
        return top_indices.tolist(), top_scores.tolist()
    
    def batch_similarity_search(self, query_embeddings: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings for multiple queries."""
        logger.info(f"Performing batch similarity search for {len(query_embeddings)} queries")
        results = []
        
        for i, query_embedding in enumerate(query_embeddings):
            indices, scores = self.similarity_search(query_embedding, embeddings, top_k)
            results.append((indices, scores))
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(query_embeddings)} queries")
        
        logger.info("Batch similarity search completed")
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if isinstance(self.model, SentenceTransformer):
            dimension = self.model.get_sentence_embedding_dimension()
            logger.debug(f"Embedding dimension (sentence-transformers): {dimension}")
            return dimension
        else:
            dimension = self.model.config.hidden_size
            logger.debug(f"Embedding dimension (transformers): {dimension}")
            return dimension
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'offload_model') and self.offload_model:
            self._unload_model() 