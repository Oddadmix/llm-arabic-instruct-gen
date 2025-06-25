"""
Command-line interface for LLM Dataset Instruction Generator.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import Settings
from processors import DocumentProcessor, TextChunker, QAGenerator, DatasetProcessor
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
        
        # Process based on input type
        if args.file:
            process_file(args.file, settings, args)
        elif args.dataset:
            process_dataset(args.dataset, settings, args)
        elif args.dataset_file:
            process_dataset_file(args.dataset_file, settings, args)
        elif args.dataset_info:
            show_dataset_info(args.dataset_info, args)
        else:
            logger.error("No input specified. Use --file, --dataset, --dataset-file, or --dataset-info option.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Dataset Instruction Generator - Generate instruction datasets from PDF, TXT documents, or Hugging Face datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDF/TXT files
  python cli.py --file document.pdf
  python cli.py --file document.txt
  python cli.py --file document.pdf --output-format json
  
  # Process Hugging Face datasets
  python cli.py --dataset "squad" --text-column "context"
  python cli.py --dataset "wikitext" --dataset-config "wikitext-103-raw-v1"
  python cli.py --dataset "arabic-news" --max-samples 1000
  
  # Process datasets with specific start index
  python cli.py --dataset "squad" --start-index 1000 --max-samples 500
  python cli.py --dataset "wikitext" --start-index 5000 --max-samples 1000
  
  # Process local dataset files
  python cli.py --dataset-file data.csv --text-column "text"
  python cli.py --dataset-file data.jsonl --file-type json
  
  # Process local files with start index
  python cli.py --dataset-file data.csv --start-index 100 --max-samples 200
  
  # Get dataset information
  python cli.py --dataset-info "squad"
  
  # General options
  python cli.py --file document.pdf --chunk-size 500 --questions-per-chunk 5
  python cli.py --dataset "squad" --llm-model Qwen/Qwen2.5-32B-Instruct --embedding-model Qwen/Qwen2-0.5B-Instruct
  python cli.py --file document.txt --max-pages 10  # Process only first 10 pages (for PDFs)
  python cli.py --file document.pdf --no-offload  # Disable model offloading to save memory
        """
    )
    
    # Input source arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to the PDF or TXT file to process"
    )
    
    input_group.add_argument(
        "--dataset",
        type=str,
        help="Hugging Face dataset name to process (e.g., 'squad', 'wikitext')"
    )
    
    input_group.add_argument(
        "--dataset-file",
        type=str,
        help="Path to local dataset file (CSV, JSON, JSONL, Parquet)"
    )
    
    input_group.add_argument(
        "--dataset-info",
        type=str,
        help="Get information about a Hugging Face dataset without processing"
    )
    
    # Dataset-specific arguments
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration name (for datasets with multiple configs) (default: None)"
    )
    
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text data (default: text)"
    )
    
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["auto", "csv", "json", "jsonl", "parquet"],
        default="auto",
        help="File type for local dataset files (default: auto-detect)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process from dataset (default: None - process all)"
    )
    
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing from in the dataset (default: 0)"
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
    
    # Document processing options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Size of text chunks (default: None - use config value)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between text chunks (default: None - use config value)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process (for PDFs only) (default: None - use config value)"
    )
    
    # QA generation options
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=None,
        help="Number of questions to generate per chunk (default: None - use config value)"
    )
    
    # LLM options
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use for question generation (default: None - use config value)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for LLM generation (default: None - use config value)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p for LLM generation (default: None - use config value)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length for LLM generation (default: None - use config value)"
    )
    
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=None,
        help="Enable sampling for LLM generation (default: None - use config value)"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model to use for text embeddings (default: None - use config value)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Device to use for models (default: None - use config value)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding generation (default: None - use config value)"
    )
    
    # Memory management options
    parser.add_argument(
        "--no-offload",
        action="store_true",
        default=None,
        help="Disable model offloading to save memory (models will stay loaded) (default: None - use config value)"
    )
    
    # Export options
    parser.add_argument(
        "--no-save-individual",
        action="store_true",
        default=False,
        help="Disable saving individual chunk and QA pair files (default: False)"
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
        default=None,
        help="Log file path (default: None - log to console only)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="LLM Dataset Instruction Generator v0.1.0"
    )
    
    return parser


def process_file(file_path: str, settings: Settings, args: argparse.Namespace):
    """Process PDF or TXT file and generate instruction dataset."""
    logger = logging.getLogger(__name__)
    
    # Validate file
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    logger.info(f"Processing file: {file_path}")
    
    # Use CLI arguments with fallback to config, then to defaults
    chunk_size = args.chunk_size if args.chunk_size is not None else getattr(settings, 'chunk_size', 1000)
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else getattr(settings, 'chunk_overlap', 200)
    max_pages = args.max_pages if args.max_pages is not None else getattr(settings, 'max_pages', None)
    questions_per_chunk = args.questions_per_chunk if args.questions_per_chunk is not None else getattr(settings, 'questions_per_chunk', 3)
    llm_model = args.llm_model if args.llm_model is not None else getattr(settings, 'llm_model', 'Qwen/Qwen2.5-32B-Instruct')
    temperature = args.temperature if args.temperature is not None else getattr(settings, 'temperature', 0.7)
    top_p = args.top_p if args.top_p is not None else getattr(settings, 'top_p', 0.9)
    max_length = args.max_length if args.max_length is not None else getattr(settings, 'max_length', 512)
    do_sample = args.do_sample if hasattr(args, 'do_sample') else getattr(settings, 'do_sample', True)
    embedding_model = args.embedding_model if args.embedding_model is not None else getattr(settings, 'embedding_model', 'Qwen/Qwen2-0.5B-Instruct')
    device = args.device if args.device is not None else getattr(settings, 'device', 'auto')
    batch_size = args.batch_size if args.batch_size is not None else getattr(settings, 'batch_size', 32)
    offload_model = not args.no_offload if hasattr(args, 'no_offload') else getattr(settings, 'offload_model', True)
    
    logger.info(f"Using parameters:")
    logger.info(f"  - Chunk size: {chunk_size}")
    logger.info(f"  - Chunk overlap: {chunk_overlap}")
    logger.info(f"  - Max pages: {max_pages}")
    logger.info(f"  - Questions per chunk: {questions_per_chunk}")
    logger.info(f"  - LLM model: {llm_model}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Top-p: {top_p}")
    logger.info(f"  - Max length: {max_length}")
    logger.info(f"  - Do sample: {do_sample}")
    logger.info(f"  - Embedding model: {embedding_model}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Offload model: {offload_model}")
    
    # Initialize processors
    doc_processor = DocumentProcessor(max_pages=max_pages)
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    qa_generator = QAGenerator(
        num_questions_per_chunk=questions_per_chunk,
        llm_model=llm_model,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        offload_model=offload_model
    )
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        offload_model=offload_model
    )
    exporter = DataExporter(output_dir=args.output_dir)
    
    try:
        # Extract text from file
        logger.info("Step 1: Extracting text from file...")
        text = doc_processor.extract_text(file_path)
        logger.info(f"Text extraction completed. Total characters: {len(text)}")
        
        # Chunk the text
        logger.info("Step 2: Chunking text...")
        text_chunks = chunker.chunk_text(text)
        logger.info(f"Text chunking completed. Total chunks: {len(text_chunks)}")
        
        # Generate QA pairs
        logger.info("Step 3: Generating QA pairs...")
        
        # Define save callback for incremental saving
        def save_incrementally(chunk, chunk_qa_pairs, chunk_index):
            if not args.no_save_individual:
                # Save chunk as separate file
                chunk_filename = f"chunk_{chunk_index:04d}.txt"
                exporter.save_chunk(chunk, chunk_filename)
                
                # Save QA pairs as separate files
                for qa_idx, qa_pair in enumerate(chunk_qa_pairs):
                    qa_filename = f"chunk_{chunk_index:04d}_qa_{qa_idx:02d}.json"
                    exporter.save_qa_pair(qa_pair, qa_filename)
        
        qa_pairs = qa_generator.generate_qa_pairs(text_chunks, save_callback=save_incrementally)
        logger.info(f"QA generation completed. Total QA pairs: {len(qa_pairs)}")
        
        # Generate embeddings
        logger.info("Step 4: Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings(text_chunks)
        logger.info(f"Embedding generation completed. Total embeddings: {len(embeddings)}")
        
        # Convert to instruction format
        logger.info("Step 5: Converting to instruction format...")
        instructions = qa_generator.generate_instruction_format(qa_pairs)
        logger.info(f"Instruction format conversion completed. Total instructions: {len(instructions)}")
        
        # Export results
        logger.info("Step 6: Exporting results...")
        exporter.export_dataset(instructions, embeddings, args.output_format)
        logger.info(f"Export completed. Results saved to: {args.output_dir}")
        
        # Print summary
        print_summary(len(text_chunks), len(qa_pairs), len(embeddings), args.output_dir)
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


def process_dataset(dataset_name: str, settings: Settings, args: argparse.Namespace):
    """Process Hugging Face dataset and generate instruction dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing Hugging Face dataset: {dataset_name}")
    
    # Use CLI arguments with fallback to config, then to defaults
    questions_per_chunk = args.questions_per_chunk if args.questions_per_chunk is not None else getattr(settings, 'questions_per_chunk', 3)
    llm_model = args.llm_model if args.llm_model is not None else getattr(settings, 'llm_model', 'Qwen/Qwen2.5-32B-Instruct')
    temperature = args.temperature if args.temperature is not None else getattr(settings, 'temperature', 0.7)
    top_p = args.top_p if args.top_p is not None else getattr(settings, 'top_p', 0.9)
    max_length = args.max_length if args.max_length is not None else getattr(settings, 'max_length', 512)
    do_sample = args.do_sample if hasattr(args, 'do_sample') else getattr(settings, 'do_sample', True)
    embedding_model = args.embedding_model if args.embedding_model is not None else getattr(settings, 'embedding_model', 'Qwen/Qwen2-0.5B-Instruct')
    device = args.device if args.device is not None else getattr(settings, 'device', 'auto')
    batch_size = args.batch_size if args.batch_size is not None else getattr(settings, 'batch_size', 32)
    offload_model = not args.no_offload if hasattr(args, 'no_offload') else getattr(settings, 'offload_model', True)
    
    logger.info(f"Using parameters:")
    logger.info(f"  - Questions per chunk: {questions_per_chunk}")
    logger.info(f"  - LLM model: {llm_model}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Top-p: {top_p}")
    logger.info(f"  - Max length: {max_length}")
    logger.info(f"  - Do sample: {do_sample}")
    logger.info(f"  - Embedding model: {embedding_model}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Offload model: {offload_model}")
    
    # Initialize processors
    dataset_processor = DatasetProcessor(
        text_column=args.text_column,
        max_samples=args.max_samples,
        start_index=args.start_index
    )
    qa_generator = QAGenerator(
        num_questions_per_chunk=questions_per_chunk,
        llm_model=llm_model,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        offload_model=offload_model
    )
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        offload_model=offload_model
    )
    exporter = DataExporter(output_dir=args.output_dir)
    
    try:
        # Load dataset and extract text chunks
        logger.info("Step 1: Loading dataset and extracting text chunks...")
        text_chunks = dataset_processor.load_dataset(
            dataset_name=dataset_name,
            split=args.dataset_split,
            config_name=args.dataset_config
        )
        logger.info(f"Dataset loading completed. Total chunks: {len(text_chunks)}")
        
        # Generate QA pairs
        logger.info("Step 2: Generating QA pairs...")
        
        # Define save callback for incremental saving
        def save_incrementally(chunk, chunk_qa_pairs, chunk_index):
            if not args.no_save_individual:
                # Save chunk as separate file
                chunk_filename = f"dataset_chunk_{chunk_index:04d}.txt"
                exporter.save_chunk(chunk, chunk_filename)
                
                # Save QA pairs as separate files
                for qa_idx, qa_pair in enumerate(chunk_qa_pairs):
                    qa_filename = f"dataset_chunk_{chunk_index:04d}_qa_{qa_idx:02d}.json"
                    exporter.save_qa_pair(qa_pair, qa_filename)
        
        qa_pairs = qa_generator.generate_qa_pairs(text_chunks, save_callback=save_incrementally)
        logger.info(f"QA generation completed. Total QA pairs: {len(qa_pairs)}")
        
        # Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings(text_chunks)
        logger.info(f"Embedding generation completed. Total embeddings: {len(embeddings)}")
        
        # Convert to instruction format
        logger.info("Step 4: Converting to instruction format...")
        instructions = qa_generator.generate_instruction_format(qa_pairs)
        logger.info(f"Instruction format conversion completed. Total instructions: {len(instructions)}")
        
        # Export results
        logger.info("Step 5: Exporting results...")
        exporter.export_dataset(instructions, embeddings, args.output_format)
        logger.info(f"Export completed. Results saved to: {args.output_dir}")
        
        # Print summary
        print_summary(len(text_chunks), len(qa_pairs), len(embeddings), args.output_dir)
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def process_dataset_file(file_path: str, settings: Settings, args: argparse.Namespace):
    """Process local dataset file and generate instruction dataset."""
    logger = logging.getLogger(__name__)
    
    # Validate file
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    logger.info(f"Processing local dataset file: {file_path}")
    
    # Use CLI arguments with fallback to config, then to defaults
    questions_per_chunk = args.questions_per_chunk if args.questions_per_chunk is not None else getattr(settings, 'questions_per_chunk', 3)
    llm_model = args.llm_model if args.llm_model is not None else getattr(settings, 'llm_model', 'Qwen/Qwen2.5-32B-Instruct')
    temperature = args.temperature if args.temperature is not None else getattr(settings, 'temperature', 0.7)
    top_p = args.top_p if args.top_p is not None else getattr(settings, 'top_p', 0.9)
    max_length = args.max_length if args.max_length is not None else getattr(settings, 'max_length', 512)
    do_sample = args.do_sample if hasattr(args, 'do_sample') else getattr(settings, 'do_sample', True)
    embedding_model = args.embedding_model if args.embedding_model is not None else getattr(settings, 'embedding_model', 'Qwen/Qwen2-0.5B-Instruct')
    device = args.device if args.device is not None else getattr(settings, 'device', 'auto')
    batch_size = args.batch_size if args.batch_size is not None else getattr(settings, 'batch_size', 32)
    offload_model = not args.no_offload if hasattr(args, 'no_offload') else getattr(settings, 'offload_model', True)
    
    logger.info(f"Using parameters:")
    logger.info(f"  - Questions per chunk: {questions_per_chunk}")
    logger.info(f"  - LLM model: {llm_model}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Top-p: {top_p}")
    logger.info(f"  - Max length: {max_length}")
    logger.info(f"  - Do sample: {do_sample}")
    logger.info(f"  - Embedding model: {embedding_model}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Offload model: {offload_model}")
    
    # Initialize processors
    dataset_processor = DatasetProcessor(
        text_column=args.text_column,
        max_samples=args.max_samples,
        start_index=args.start_index
    )
    qa_generator = QAGenerator(
        num_questions_per_chunk=questions_per_chunk,
        llm_model=llm_model,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        offload_model=offload_model
    )
    embedding_generator = EmbeddingGenerator(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        offload_model=offload_model
    )
    exporter = DataExporter(output_dir=args.output_dir)
    
    try:
        # Load dataset and extract text chunks
        logger.info("Step 1: Loading dataset file and extracting text chunks...")
        text_chunks = dataset_processor.load_dataset_from_file(
            file_path=file_path,
            file_type=args.file_type
        )
        logger.info(f"Dataset loading completed. Total chunks: {len(text_chunks)}")
        
        # Generate QA pairs
        logger.info("Step 2: Generating QA pairs...")
        
        # Define save callback for incremental saving
        def save_incrementally(chunk, chunk_qa_pairs, chunk_index):
            if not args.no_save_individual:
                # Save chunk as separate file
                chunk_filename = f"file_chunk_{chunk_index:04d}.txt"
                exporter.save_chunk(chunk, chunk_filename)
                
                # Save QA pairs as separate files
                for qa_idx, qa_pair in enumerate(chunk_qa_pairs):
                    qa_filename = f"file_chunk_{chunk_index:04d}_qa_{qa_idx:02d}.json"
                    exporter.save_qa_pair(qa_pair, qa_filename)
        
        qa_pairs = qa_generator.generate_qa_pairs(text_chunks, save_callback=save_incrementally)
        logger.info(f"QA generation completed. Total QA pairs: {len(qa_pairs)}")
        
        # Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings(text_chunks)
        logger.info(f"Embedding generation completed. Total embeddings: {len(embeddings)}")
        
        # Convert to instruction format
        logger.info("Step 4: Converting to instruction format...")
        instructions = qa_generator.generate_instruction_format(qa_pairs)
        logger.info(f"Instruction format conversion completed. Total instructions: {len(instructions)}")
        
        # Export results
        logger.info("Step 5: Exporting results...")
        exporter.export_dataset(instructions, embeddings, args.output_format)
        logger.info(f"Export completed. Results saved to: {args.output_dir}")
        
        # Print summary
        print_summary(len(text_chunks), len(qa_pairs), len(embeddings), args.output_dir)
        
    except Exception as e:
        logger.error(f"Error processing dataset file: {e}")
        raise


def show_dataset_info(dataset_name: str, args: argparse.Namespace):
    """Show information about a Hugging Face dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Getting information for dataset: {dataset_name}")
    
    # Initialize dataset processor
    dataset_processor = DatasetProcessor(
        text_column=args.text_column,
        max_samples=args.max_samples
    )
    
    try:
        # Get dataset info
        info = dataset_processor.get_dataset_info(
            dataset_name=dataset_name,
            split=args.dataset_split,
            config_name=args.dataset_config
        )
        
        # Print dataset information
        print("\n" + "="*60)
        print(f"DATASET INFORMATION: {info['name']}")
        print("="*60)
        print(f"Split: {info['split']}")
        if info['config']:
            print(f"Config: {info['config']}")
        print(f"Total samples: {info['total_samples']:,}")
        print(f"Columns: {', '.join(info['columns'])}")
        print(f"Text column '{args.text_column}' exists: {info['text_column_exists']}")
        
        if 'sample_data' in info:
            print(f"\nSample data:")
            for key, value in info['sample_data'].items():
                print(f"  {key}: {value}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise


def print_summary(num_chunks: int, num_qa_pairs: int, num_embeddings: int, output_dir: str):
    """Print processing summary."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Text chunks processed: {num_chunks:,}")
    print(f"QA pairs generated: {num_qa_pairs:,}")
    print(f"Embeddings generated: {num_embeddings:,}")
    print(f"Average QA pairs per chunk: {num_qa_pairs / num_chunks:.1f}")
    print(f"Output directory: {output_dir}")
    print("="*60)


# Keep the old function name for backward compatibility
def process_pdf(pdf_file: str, settings: Settings, args: argparse.Namespace):
    """Alias for process_file for backward compatibility."""
    return process_file(pdf_file, settings, args)


if __name__ == "__main__":
    main() 