"""
Main application module for LLM Dataset Instruction Generator.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from config import Settings
from processors import PDFProcessor, TextChunker, QAGenerator
from utils import DataExporter, EmbeddingGenerator, setup_colored_logging


class LLMDatasetGenerator:
    """Main application class for generating LLM instruction datasets."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the dataset generator."""
        self.settings = Settings(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pdf_processor = None
        self.chunker = None
        self.qa_generator = None
        self.exporter = None
        self.embedding_generator = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components."""
        # PDF Processor
        self.pdf_processor = PDFProcessor(
            max_pages=self.settings.get("pdf_processor.max_pages")
        )
        
        # Text Chunker
        self.chunker = TextChunker(
            chunk_size=self.settings.get("pdf_processor.chunk_size"),
            chunk_overlap=self.settings.get("pdf_processor.chunk_overlap")
        )
        
        # QA Generator with LLM configuration
        self.qa_generator = QAGenerator(
            num_questions_per_chunk=self.settings.get("qa_generator.num_questions_per_chunk"),
            question_types=self.settings.get("qa_generator.question_types"),
            llm_model=self.settings.get("qa_generator.llm_model"),
            max_length=self.settings.get("qa_generator.max_length"),
            temperature=self.settings.get("qa_generator.temperature"),
            top_p=self.settings.get("qa_generator.top_p"),
            do_sample=self.settings.get("qa_generator.do_sample")
        )
        
        # Data Exporter
        self.exporter = DataExporter(
            output_dir=self.settings.get("export.output_dir")
        )
        
        # Embedding Generator with device configuration
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.settings.get("embeddings.model"),
            batch_size=self.settings.get("embeddings.batch_size"),
            device=self.settings.get("embeddings.device")
        )
    
    def process_pdf(self, pdf_path: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF file and generate instruction dataset."""
        self.logger.info(f"Starting processing of PDF: {pdf_path}")
        
        # Validate input
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text from PDF
        self.logger.info("Extracting text from PDF...")
        text = self.pdf_processor.extract_text(pdf_path)
        
        if not text.strip():
            raise ValueError("No text extracted from PDF")
        
        # Chunk the text
        self.logger.info("Chunking text...")
        chunks = self.chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("No chunks created from text")
        
        self.logger.info(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings (optional)
        self.logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        # Generate QA pairs
        self.logger.info("Generating question-answer pairs...")
        qa_pairs = self.qa_generator.generate_qa_pairs(chunks)
        
        if not qa_pairs:
            raise ValueError("No QA pairs generated")
        
        self.logger.info(f"Generated {len(qa_pairs)} QA pairs")
        
        # Export dataset
        self.logger.info("Exporting dataset...")
        base_filename = output_filename or pdf_file.stem
        
        exported_files = self.exporter.export_multiple_formats(qa_pairs, base_filename, chunks=chunks)
        
        # Create metadata
        metadata = self.exporter.create_metadata(qa_pairs, pdf_path)
        metadata_file = self.exporter.export_json([metadata], f"{base_filename}_metadata")
        
        # Prepare results
        results = {
            "pdf_file": pdf_path,
            "text_chunks": len(chunks),
            "qa_pairs": len(qa_pairs),
            "exported_files": exported_files,
            "metadata_file": metadata_file,
            "embeddings_shape": embeddings.shape if len(embeddings) > 0 else None
        }
        
        self.logger.info("Processing completed successfully!")
        return results
    
    def process_multiple_pdfs(self, pdf_paths: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple PDF files and generate combined dataset."""
        self.logger.info(f"Processing {len(pdf_paths)} PDF files")
        
        all_qa_pairs = []
        results = {
            "processed_files": [],
            "total_qa_pairs": 0,
            "exported_files": {}
        }
        
        for pdf_path in pdf_paths:
            try:
                file_result = self.process_pdf(pdf_path)
                all_qa_pairs.extend(file_result.get("qa_pairs", []))
                results["processed_files"].append(file_result)
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        if all_qa_pairs:
            # Export combined dataset
            combined_filename = "combined_dataset"
            exported_files = self.exporter.export_multiple_formats(all_qa_pairs, combined_filename)
            
            results["total_qa_pairs"] = len(all_qa_pairs)
            results["exported_files"] = exported_files
            
            self.logger.info(f"Combined dataset with {len(all_qa_pairs)} QA pairs exported")
        
        return results
    
    def get_statistics(self, qa_pairs: list) -> Dict[str, Any]:
        """Get statistics about generated QA pairs."""
        if not qa_pairs:
            return {}
        
        stats = {
            "total_pairs": len(qa_pairs),
            "question_types": {},
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "avg_context_length": 0
        }
        
        question_lengths = []
        answer_lengths = []
        context_lengths = []
        
        for qa in qa_pairs:
            question_lengths.append(len(qa.get("question", "")))
            answer_lengths.append(len(qa.get("answer", "")))
            context_lengths.append(len(qa.get("context", "")))
            
            q_type = qa.get("question_type", "unknown")
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1
        
        if question_lengths:
            stats["avg_question_length"] = sum(question_lengths) / len(question_lengths)
            stats["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
            stats["avg_context_length"] = sum(context_lengths) / len(context_lengths)
        
        return stats


def main():
    """Main entry point for the application."""
    # Setup logging
    setup_colored_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize generator
        generator = LLMDatasetGenerator()
        
        # Example usage
        logger.info("LLM Dataset Instruction Generator initialized")
        logger.info("Use the CLI interface (python cli.py) or import this module for programmatic usage")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise


if __name__ == "__main__":
    main() 