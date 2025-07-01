"""
Settings and configuration management for the LLM Dataset Instruction Generator.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Settings:
    """Application settings and configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/default_config.json"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "pdf_processor": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_pages": None
            },
            "qa_generator": {
                "num_questions_per_chunk": 3,
                "question_types": ["what", "how", "why"],
                "max_answer_length": 200,
                "llm_model": "microsoft/DialoGPT-medium",
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "offload_model": True,
                "llm_backend": "transformers",
                "openai_api_key": None,
                "openai_api_base": None
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "auto"
            },
            "export": {
                "format": "json",
                "output_dir": "output"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2) 