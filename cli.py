"""
Command-line interface for LLM Dataset Instruction Generator.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import Settings
from processors import PDFProcessor, TextChunker, QAGenerator
from utils import DataExporter, setup_colored_logging, EmbeddingGenerator
import logging


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_colored_logging(
        level=args.log_level,
        log_file=args.log_file
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        settings = Settings(args.config)
        
        # Process PDF
        if args.pdf_file:
            process_pdf(args.pdf_file, settings, args)
        else:
            logger.error("No PDF file specified. Use --pdf-file option.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Dataset Instruction Generator - Generate instruction datasets from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --pdf-file document.pdf
  python cli.py --pdf-file document.pdf --output-format json
  python cli.py --pdf-file document.pdf --chunk-size 500 --questions-per-chunk 5
  python cli.py --pdf-file document.pdf --llm-model Qwen/Qwen2.5-32B-Instruct --embedding-model Qwen/Qwen2-0.5B-Instruct
  python cli.py --pdf-file document.pdf --max-pages 10  # Process only first 10 pages
  python cli.py --pdf-file document.pdf --no-offload  # Disable model offloading to save memory
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--pdf-file",
        type=str,
        help="Path to the PDF file to process"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file (default: config/default_config.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated files (default: output)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "all"],
        default="all",
        help="Output format for generated dataset (default: all)"
    )
    
    # PDF processing options
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Size of text chunks (overrides config)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Overlap between text chunks (overrides config)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process (overrides config)"
    )
    
    # QA generation options
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        help="Number of questions to generate per chunk (overrides config)"
    )
    
    # LLM options
    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM model to use for question generation (overrides config)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM generation (overrides config)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p for LLM generation (overrides config)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum length for LLM generation (overrides config)"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model to use for text embeddings (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for models (overrides config)"
    )
    
    # Memory management options
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Disable model offloading to save memory (models will stay loaded)"
    )
    
    # Export options
    parser.add_argument(
        "--no-save-individual",
        action="store_true",
        help="Disable saving individual chunk and QA pair files"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (optional)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="LLM Dataset Instruction Generator v0.1.0"
    )
    
    return parser


def process_pdf(pdf_file: str, settings: Settings, args: argparse.Namespace):
    """Process PDF file and generate instruction dataset."""
    logger = logging.getLogger(__name__)
    
    # Validate PDF file
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_file}")
        return
    
    logger.info(f"Processing PDF: {pdf_file}")
    
    # Determine model offloading setting
    offload_models = not args.no_offload
    logger.info(f"Model offloading: {'enabled' if offload_models else 'disabled'}")
    
    # Initialize processors with settings
    pdf_processor = PDFProcessor(
        max_pages=args.max_pages or settings.get("pdf_processor.max_pages")
    )
    
    chunker = TextChunker(
        chunk_size=args.chunk_size or settings.get("pdf_processor.chunk_size"),
        chunk_overlap=args.chunk_overlap or settings.get("pdf_processor.chunk_overlap")
    )
    
    # QA Generator with LLM configuration
    qa_generator = QAGenerator(
        num_questions_per_chunk=args.questions_per_chunk or settings.get("qa_generator.num_questions_per_chunk"),
        question_types=settings.get("qa_generator.question_types"),
        llm_model=args.llm_model or settings.get("qa_generator.llm_model"),
        max_length=args.max_length or settings.get("qa_generator.max_length"),
        temperature=args.temperature or settings.get("qa_generator.temperature"),
        top_p=args.top_p or settings.get("qa_generator.top_p"),
        do_sample=settings.get("qa_generator.do_sample"),
        offload_model=offload_models
    )
    
    # Embedding generator with model override
    embedding_model = args.embedding_model or settings.get("embeddings.model")
    embedding_device = args.device or settings.get("embeddings.device")
    embedding_batch_size = settings.get("embeddings.batch_size")
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=embedding_device,
        offload_model=offload_models
    )
    
    exporter = DataExporter(output_dir=args.output_dir)
    
    # Extract text from PDF
    logger.info("Extracting text from PDF...")
    text = pdf_processor.extract_text(pdf_file)
    
    if not text.strip():
        logger.warning("No text extracted from PDF")
        return
    
    # Chunk the text
    logger.info("Chunking text...")
    chunks = chunker.chunk_text(text)
    
    if not chunks:
        logger.warning("No chunks created from text")
        return
    
    logger.info(f"Created {len(chunks)} text chunks")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.generate_embeddings(chunks)
    
    # Export dataset
    logger.info("Exporting dataset...")
    base_filename = pdf_path.stem
    
    # Determine export settings - by default, save individual files unless --no-save-individual is used
    save_individual_files = not args.no_save_individual
    
    # Generate QA pairs
    logger.info("Generating question-answer pairs...")
    
    # Create callback function for incremental saving
    def save_incrementally(chunk, chunk_qa_pairs, chunk_index):
        """Callback function to save chunk and QA pairs incrementally."""
        if save_individual_files:
            # Save chunk
            exporter.save_chunk_incrementally(chunk, chunk_index, base_filename)
            # Save QA pairs for this chunk
            exporter.save_qa_pairs_incrementally(chunk_qa_pairs, chunk_index, base_filename)
    
    qa_pairs = qa_generator.generate_qa_pairs(chunks, save_callback=save_incrementally if save_individual_files else None)
    
    if not qa_pairs:
        logger.warning("No QA pairs generated")
        return
    
    logger.info(f"Generated {len(qa_pairs)} QA pairs")
    
    # Note: Individual files are already saved incrementally, so we only need to export the combined formats
    if args.output_format == "all":
        exported_files = exporter.export_multiple_formats(qa_pairs, base_filename)
        logger.info(f"Exported dataset in multiple formats:")
        for format_name, filepath in exported_files.items():
            logger.info(f"  {format_name}: {filepath}")
    else:
        if args.output_format == "json":
            filepath = exporter.export_json(qa_pairs, base_filename)
        elif args.output_format == "csv":
            filepath = exporter.export_csv(qa_pairs, base_filename)
        logger.info(f"Exported dataset to: {filepath}")
    
    # Create and export metadata
    metadata = exporter.create_metadata(qa_pairs, pdf_file)
    metadata_file = exporter.export_json([metadata], f"{base_filename}_metadata")
    logger.info(f"Exported metadata to: {metadata_file}")
    
    # Summary of saved files
    if save_individual_files:
        chunks_dir = exporter.output_dir / f"{base_filename}_chunks"
        qa_pairs_dir = exporter.output_dir / f"{base_filename}_qa_pairs"
        logger.info("=== INCREMENTAL SAVING SUMMARY ===")
        logger.info(f"Individual chunks saved to: {chunks_dir}")
        logger.info(f"Individual QA pairs saved to: {qa_pairs_dir}")
        logger.info(f"Total chunks processed: {len(chunks)}")
        logger.info(f"Total QA pairs generated: {len(qa_pairs)}")
        logger.info("Files were saved incrementally during processing")
    
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main() 