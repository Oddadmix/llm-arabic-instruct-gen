"""
Data export utilities for saving generated datasets.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles export of generated datasets in various formats."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_json(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Export data to JSON format."""
        filepath = self.output_dir / f"{filename}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(data)} items to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    def export_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Export data to CSV format."""
        filepath = self.output_dir / f"{filename}.csv"
        
        if not data:
            logger.warning("No data to export")
            return str(filepath)
        
        try:
            # Get all unique keys from all dictionaries
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Exported {len(data)} items to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def export_instruction_format(self, qa_pairs: List[Dict[str, Any]], filename: str) -> str:
        """Export QA pairs in instruction format for training."""
        instructions = []
        
        for qa in qa_pairs:
            instruction = {
                "instruction": qa["question"],
                "input": qa["context"],
                "output": qa["answer"]
            }
            instructions.append(instruction)
        
        return self.export_json(instructions, f"{filename}_instructions")
    
    def export_alpaca_format(self, qa_pairs: List[Dict[str, Any]], filename: str) -> str:
        """Export in Alpaca format for training (single JSON file)."""
        alpaca_data = []
        
        for qa in qa_pairs:
            alpaca_item = {
                "instruction": qa["question"],
                "input": qa["context"],
                "output": qa["answer"]
            }
            alpaca_data.append(alpaca_item)
        
        return self.export_json(alpaca_data, f"{filename}_alpaca")

    def export_chunks_as_txt(self, chunks: List[str], base_filename: str) -> str:
        """Export each chunk as a separate .txt file in a subfolder."""
        chunk_dir = self.output_dir / f"{base_filename}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, chunk in enumerate(chunks):
            # Use zero-padded 4-digit format for chunk filenames
            chunk_path = chunk_dir / f"chunk_{idx:04d}.txt"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        logger.info(f"Exported {len(chunks)} chunks as .txt files to {chunk_dir}")
        return str(chunk_dir)

    def export_alpaca_datapoints_as_json(self, qa_pairs: List[Dict[str, Any]], base_filename: str) -> str:
        """Export each Alpaca-format data point as a separate .json file in a subfolder."""
        alpaca_dir = self.output_dir / f"{base_filename}_alpaca"
        alpaca_dir.mkdir(parents=True, exist_ok=True)
        for idx, qa in enumerate(qa_pairs):
            alpaca_item = {
                "instruction": qa["question"],
                "input": qa["context"],
                "output": qa["answer"]
            }
            datapoint_path = alpaca_dir / f"datapoint_{idx}.json"
            with open(datapoint_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_item, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(qa_pairs)} Alpaca data points as .json files to {alpaca_dir}")
        return str(alpaca_dir)

    def export_chat_format(self, qa_pairs: List[Dict[str, Any]], filename: str) -> str:
        """Export in chat format for training."""
        chat_data = []
        
        for qa in qa_pairs:
            chat_item = {
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            }
            chat_data.append(chat_item)
        
        return self.export_json(chat_data, f"{filename}_chat")
    
    def export_multiple_formats(self, qa_pairs: List[Dict[str, Any]], base_filename: str) -> Dict[str, str]:
        """Export data in multiple formats, including per-datapoint files."""
        exported_files = {}
        
        # Export raw QA pairs
        exported_files["json"] = self.export_json(qa_pairs, base_filename)
        exported_files["csv"] = self.export_csv(qa_pairs, base_filename)
        
        # Export in different training formats
        exported_files["instructions"] = self.export_instruction_format(qa_pairs, base_filename)
        exported_files["alpaca"] = self.export_alpaca_format(qa_pairs, base_filename)
        exported_files["alpaca_datapoints"] = self.export_alpaca_datapoints_as_json(qa_pairs, base_filename)
        exported_files["chat"] = self.export_chat_format(qa_pairs, base_filename)
        
        logger.info(f"Exported data in {len(exported_files)} formats")
        return exported_files
    
    def create_metadata(self, qa_pairs: List[Dict[str, Any]], source_file: str) -> Dict[str, Any]:
        """Create metadata for the exported dataset."""
        metadata = {
            "dataset_info": {
                "name": f"Generated from {Path(source_file).stem}",
                "description": "Instruction dataset generated from PDF document",
                "source_file": source_file,
                "total_qa_pairs": len(qa_pairs),
                "generation_timestamp": str(Path().cwd().stat().st_mtime)
            },
            "statistics": {
                "question_types": {},
                "avg_question_length": 0,
                "avg_answer_length": 0
            }
        }
        
        # Calculate statistics
        question_lengths = []
        answer_lengths = []
        
        for qa in qa_pairs:
            question_lengths.append(len(qa["question"]))
            answer_lengths.append(len(qa["answer"]))
            
            q_type = qa.get("question_type", "unknown")
            metadata["statistics"]["question_types"][q_type] = metadata["statistics"]["question_types"].get(q_type, 0) + 1
        
        if question_lengths:
            metadata["statistics"]["avg_question_length"] = sum(question_lengths) / len(question_lengths)
            metadata["statistics"]["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
        
        return metadata

    def export_qa_pairs_as_json(self, qa_pairs: List[Dict[str, Any]], base_filename: str) -> str:
        """Export each QA pair as a separate .json file in a subfolder."""
        qa_pairs_dir = self.output_dir / f"{base_filename}_qa_pairs"
        qa_pairs_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, qa_pair in enumerate(qa_pairs):
            # Use zero-padded 4-digit format for QA pair filenames
            qa_file = qa_pairs_dir / f"qa_pair_{idx:04d}.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pair, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(qa_pairs)} QA pairs as .json files to {qa_pairs_dir}")
        return str(qa_pairs_dir)

    def save_chunk_incrementally(self, chunk: str, chunk_index: int, base_filename: str) -> str:
        """Save a single chunk as a .txt file."""
        chunk_dir = self.output_dir / f"{base_filename}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.txt"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        
        logger.debug(f"Saved chunk {chunk_index} to {chunk_path}")
        return str(chunk_path)

    def save_qa_pairs_incrementally(self, qa_pairs: List[Dict[str, Any]], chunk_index: int, base_filename: str) -> str:
        """Save QA pairs for a chunk as separate .json files."""
        qa_dir = self.output_dir / f"{base_filename}_qa_pairs"
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for qa_idx, qa_pair in enumerate(qa_pairs):
            qa_path = qa_dir / f"chunk_{chunk_index:04d}_qa_{qa_idx:02d}.json"
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pair, f, indent=2, ensure_ascii=False)
            saved_files.append(str(qa_path))
        
        logger.debug(f"Saved {len(qa_pairs)} QA pairs for chunk {chunk_index}")
        return str(qa_dir)

    def save_chunk(self, chunk: str, filename: str) -> str:
        """Save a single chunk as a .txt file with the given filename."""
        chunk_path = self.output_dir / filename
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        
        logger.debug(f"Saved chunk to {chunk_path}")
        return str(chunk_path)

    def save_qa_pair(self, qa_pair: Dict[str, Any], filename: str) -> str:
        """Save a single QA pair as a .json file with the given filename."""
        qa_path = self.output_dir / filename
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pair, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved QA pair to {qa_path}")
        return str(qa_path)

    def export_dataset(self, instructions: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None, 
                      output_format: str = "all") -> Dict[str, str]:
        """Export the complete dataset in the specified format(s)."""
        exported_files = {}
        
        # Generate base filename from timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"dataset_{timestamp}"
        
        if output_format in ["json", "all"]:
            exported_files["instructions_json"] = self.export_json(instructions, f"{base_filename}_instructions")
        
        if output_format in ["csv", "all"]:
            exported_files["instructions_csv"] = self.export_csv(instructions, f"{base_filename}_instructions")
        
        if output_format == "all":
            # Export in multiple formats
            exported_files.update(self.export_multiple_formats(instructions, base_filename))
        
        # Save embeddings if provided
        if embeddings is not None:
            embeddings_file = self.output_dir / f"{base_filename}_embeddings.npy"
            np.save(embeddings_file, embeddings)
            exported_files["embeddings"] = str(embeddings_file)
            logger.info(f"Saved embeddings to {embeddings_file}")
        
        logger.info(f"Dataset exported successfully. Files: {list(exported_files.keys())}")
        return exported_files 