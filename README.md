# LLM Dataset Instruction Generator

A comprehensive tool for generating instruction datasets from PDF and TXT documents, or Hugging Face datasets using Large Language Models (LLMs). This tool extracts text from PDFs, TXT files, or pre-chunked datasets, generates question-answer pairs using LLMs, and creates embeddings for similarity search.

## Features

- **Multi-Format Document Support**: Extract text from PDF documents and TXT files
- **Hugging Face Dataset Support**: Process pre-chunked text from Hugging Face datasets
- **Local Dataset File Support**: Process CSV, JSON, JSONL, and Parquet files
- **Intelligent Text Chunking**: Split text into optimal chunks with configurable overlap (for files)
- **Pre-chunked Data Processing**: Use datasets with pre-chunked text data
- **Self-Contained Q&A Generation**: Generate question-answer pairs where questions include all necessary context, making them independent of the original source text
- **Two-Step LLM-Powered QA Generation**: Generate high-quality question-answer pairs using a two-step process:
  1. **Question Generation**: Generate self-contained questions that include all relevant details, names, dates, and context needed to answer them
  2. **Answer Generation**: Generate comprehensive answers based on the question and context
- **Embedding Generation**: Create vector embeddings for semantic search and similarity matching
- **Multiple Export Formats**: Export data in JSON, CSV, and Alpaca instruction formats
- **Memory Management**: Automatic model offloading to save memory during processing
- **Arabic Language Support**: Full support for Arabic text processing and chunking
- **Detailed Logging**: Comprehensive logging throughout the pipeline
- **Configurable Pipeline**: Flexible configuration via JSON files and command-line options

## Q&A Generation Approach

### Self-Contained Questions

The tool generates **self-contained questions** that include all necessary information to answer them without requiring the original context. This approach creates traditional Q&A pairs rather than RAG (Retrieval-Augmented Generation) style questions.

#### Key Characteristics:

1. **Complete Context**: Questions include all relevant details, names, dates, facts, and background information
2. **Standalone**: Questions can be answered by someone who hasn't read the original text
3. **Natural Language**: Questions are written in natural, conversational Arabic
4. **Diverse Types**: Covers factual, analytical, comparative, causal, hypothetical, and evaluative questions

#### Examples:

**Good Self-Contained Questions:**
- "ما هي العاصمة السياسية والاقتصادية لمصر؟" (What is the political and economic capital of Egypt?)
- "كيف يؤثر تغير المناخ على الزراعة في منطقة الشرق الأوسط؟" (How does climate change affect agriculture in the Middle East?)
- "ما هي الفوائد الصحية للتمر على جسم الإنسان؟" (What are the health benefits of dates on the human body?)

**Comprehensive Answers:**
- Detailed, well-structured responses
- Include relevant examples and explanations
- Written in clear, natural Arabic
- Provide complete information to satisfy the question

### Output Format

The generated Q&A pairs are exported in instruction format with empty input fields:

```json
{
  "instruction": "ما هي العاصمة السياسية والاقتصادية لمصر؟",
  "input": "",
  "output": "القاهرة هي العاصمة السياسية والاقتصادية لمصر...",
  "id": "chunk_0_question_0"
}
```

This format is ideal for training instruction-following models where the question itself contains all necessary context.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-dataset-instruct-gen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Processing PDF/TXT Files

Process a PDF file with default settings:

```bash
python cli.py --file books/sample.pdf
```

Process a TXT file with default settings:

```bash
python cli.py --file books/sample.txt
```

Process only the first page for testing (PDF only):

```bash
python cli.py --file books/sample.pdf --max-pages 1
```

### Processing Hugging Face Datasets

Process a Hugging Face dataset with default settings:

```bash
python cli.py --dataset "squad" --text-column "context"
```

Process a dataset with specific configuration:

```bash
python cli.py --dataset "wikitext" --dataset-config "wikitext-103-raw-v1" --text-column "text"
```

Process only a subset of samples:

```bash
python cli.py --dataset "arabic-news" --max-samples 1000 --text-column "content"
```

### Processing Local Dataset Files

Process a CSV file:

```bash
python cli.py --dataset-file data.csv --text-column "text"
```

Process a JSONL file:

```bash
python cli.py --dataset-file data.jsonl --file-type json --text-column "content"
```

Process a Parquet file:

```bash
python cli.py --dataset-file data.parquet --file-type parquet --text-column "text"
```

### Getting Dataset Information

Get information about a Hugging Face dataset without processing:

```bash
python cli.py --dataset-info "squad"
```

### Advanced Usage

Process with custom parameters:

```bash
python cli.py --file books/sample.pdf \
    --chunk-size 800 \
    --questions-per-chunk 5 \
    --llm-model "Qwen/Qwen2.5-7B-Instruct" \
    --embedding-model "Qwen/Qwen2-0.5B-Instruct" \
    --output-dir my_output
```

Process dataset with custom parameters:

```bash
python cli.py --dataset "squad" \
    --text-column "context" \
    --questions-per-chunk 3 \
    --llm-model "Qwen/Qwen2.5-7B-Instruct" \
    --max-samples 500
```

Process TXT file with custom parameters:

```bash
python cli.py --file books/sample.txt \
    --chunk-size 600 \
    --questions-per-chunk 3 \
    --llm-model "Qwen/Qwen2.5-7B-Instruct"
```

Process with explicit model type specification:

```bash
# For instruction-based models (even if name doesn't contain "instruct")
python cli.py --file books/sample.pdf \
    --llm-model "Qwen/Qwen2.5-7B" \
    --model-type instruction

# For completion/prompt-based models
python cli.py --file books/sample.txt \
    --llm-model "Qwen/Qwen2.5-7B" \
    --model-type completion

# Auto-detect model type (default)
python cli.py --file books/sample.pdf \
    --llm-model "Qwen/Qwen2.5-7B-Instruct" \
    --model-type auto
```

Process only the first few pages of a large PDF:

```bash
# Process only the first 10 pages
python cli.py --file books/large_document.pdf --max-pages 10

# Process only the first 5 pages with custom chunk size
python cli.py --file books/large_document.pdf --max-pages 5 --chunk-size 500

# Process only first 3 pages for quick testing with smaller models
python cli.py --file books/large_document.pdf --max-pages 3 --llm-model "Qwen/Qwen2.5-7B-Instruct"
```

### Export Control

Control which files are saved:

```bash
# Save only the main output files (no individual chunk/QA files)
python cli.py --file books/sample.pdf --no-save-individual

# Default behavior - saves all files including individual chunks and QA pairs
python cli.py --file books/sample.txt
```

### Memory Management

The tool includes automatic memory management to handle large models efficiently:

```bash
# Enable model offloading (default) - saves memory by unloading models when not in use
python cli.py --file books/sample.pdf

# Disable model offloading - keeps models loaded for faster processing
python cli.py --file books/sample.txt --no-offload
```

## Configuration

### Default Configuration

The tool uses a default configuration file (`config/default_config.json`) with the following structure:

```json
{
  "pdf_processor": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_pages": null
  },
  "qa_generator": {
    "num_questions_per_chunk": 3,
    "max_answer_length": 200,
    "llm_model": "Qwen/Qwen3-8B",
    "model_type": "auto",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true,
    "offload_model": true
  },
  "embeddings": {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "batch_size": 32,
    "device": "auto",
    "offload_model": true
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
```

### Custom Configuration

Create a custom configuration file and use it:

```bash
python cli.py --file books/sample.pdf --config my_config.json
python cli.py --file books/sample.txt --config my_config.json
python cli.py --dataset "squad" --config my_config.json
```

## Command Line Options

### Input Source Options (Mutually Exclusive)

- `--file`: Path to the PDF or TXT file to process
- `--dataset`: Hugging Face dataset name to process (e.g., 'squad', 'wikitext')
- `--dataset-file`: Path to local dataset file (CSV, JSON, JSONL, Parquet)
- `--dataset-info`: Get information about a Hugging Face dataset without processing

### Dataset-Specific Options

- `--dataset-config`: Dataset configuration name (for datasets with multiple configs) (default: None)
- `--dataset-split`: Dataset split to use (default: train)
- `--text-column`: Column name containing text data (default: text)
- `--file-type`: File type for local dataset files (auto, csv, json, jsonl, parquet) (default: auto-detect)
- `--max-samples`: Maximum number of samples to process from dataset (default: None - process all)

### Main Options

- `--config`: Path to configuration file (default: config/default_config.json)
- `--output-dir`: Output directory for generated files (default: output)
- `--output-format`: Output format (json, csv, all) (default: all)

### Document Processing Options (for files only)

- `--chunk-size`: Size of text chunks (default: None - use config value, fallback: 1000)
- `--chunk-overlap`: Overlap between text chunks (default: None - use config value, fallback: 200)
- `--max-pages`: Maximum number of pages to process (for PDFs only) (default: None - use config value)

### QA Generation Options

- `--questions-per-chunk`: Number of questions to generate per chunk (default: None - use config value, fallback: 3)
- `--llm-model`: LLM model for question generation (default: None - use config value, fallback: Qwen/Qwen2.5-32B-Instruct)
- `--temperature`: Temperature for LLM generation (default: None - use config value, fallback: 0.7)
- `--top-p`: Top-p for LLM generation (default: None - use config value, fallback: 0.9)
- `--max-length`: Maximum length for LLM generation (default: None - use config value, fallback: 512)
- `--do-sample`: Enable sampling for LLM generation (default: None - use config value, fallback: True)

### Embedding Options

- `--embedding-model`: Embedding model for text embeddings (default: None - use config value, fallback: Qwen/Qwen2-0.5B-Instruct)
- `--device`: Device to use for models (cpu, cuda, mps, auto) (default: None - use config value, fallback: auto)
- `--batch-size`: Batch size for embedding generation (default: None - use config value, fallback: 32)

### Memory Management Options

- `--no-offload`: Disable model offloading to save memory (models will stay loaded) (default: None - use config value, fallback: False)

### Export Options

- `--no-save-individual`: Disable saving individual chunk and QA pair files (default: None - use config value, fallback: False)

### Logging Options

- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `--log-file`: Log file path (default: None - log to console only)

## Parameter Hierarchy

The tool uses a three-tier parameter hierarchy:

1. **CLI Arguments**: Highest priority - explicitly specified command-line arguments
2. **Config File**: Medium priority - values from the configuration file
3. **Built-in Defaults**: Lowest priority - hardcoded fallback values

For example, if you specify `--llm-model "custom-model"`, it will use that model regardless of what's in the config file. If you don't specify the argument, it will use the value from the config file, and if that's not available, it will use the built-in default.

## Quick Examples with Defaults

### Minimal Commands (using all defaults):

```bash
# Process PDF with all defaults
python cli.py --file document.pdf

# Process dataset with all defaults
python cli.py --dataset "squad"

# Process local file with all defaults
python cli.py --dataset-file data.csv
```

### Custom Parameters (overriding defaults):

```bash
# Override specific parameters
python cli.py --file document.pdf --chunk-size 800 --questions-per-chunk 5

# Override model parameters
python cli.py --dataset "squad" --llm-model "Qwen/Qwen2.5-7B-Instruct" --temperature 0.8

# Override memory settings
python cli.py --file document.pdf --no-offload --batch-size 64
```

## Output Structure

The tool generates the following output structure:

```
output/
├── sample_qa_pairs.json          # QA pairs in JSON format
├── sample_instructions.json      # Alpaca instruction format
├── sample_chunks.txt            # Individual text chunks
├── sample_embeddings.npy        # Vector embeddings
├── sample_metadata.json         # Processing metadata
├── chunks/                      # Individual chunk files (saved incrementally)
│   ├── chunk_0001.txt
│   ├── chunk_0002.txt
│   └── ...
└── qa_pairs/                    # Individual QA pair files (saved incrementally)
    ├── qa_pair_0001_0001.json
    ├── qa_pair_0001_0002.json
    └── ...
```

**Note**: Individual chunk and QA pair files are saved incrementally during processing, so they are available for inspection even if the process is interrupted.

## Memory Management Features

### Automatic Model Offloading

The tool includes intelligent memory management to handle large transformer models:

1. **Lazy Loading**: Models are only loaded when needed
2. **Automatic Unloading**: Models are unloaded after use to free memory
3. **Periodic Cleanup**: Models are unloaded periodically during long processing runs
4. **GPU Memory Management**: CUDA cache is cleared when models are unloaded
5. **Garbage Collection**: Forced garbage collection after model unloading

### Memory Usage Optimization

- **Default Behavior**: Models are offloaded by default to save memory
- **Performance Mode**: Use `--no-offload` to keep models loaded for faster processing
- **Batch Processing**: Embeddings are generated in configurable batches
- **Progress Logging**: Memory usage and model loading states are logged

### Incremental File Saving

The tool saves files incrementally during processing to prevent data loss:

- **Chunk Files**: Each text chunk is saved as a separate .txt file immediately after processing
- **QA Pair Files**: Each QA pair is saved as a separate .json file immediately after generation
- **Progress Recovery**: If processing is interrupted, you can resume from where you left off
- **Real-time Monitoring**: Files are available for inspection as they are generated

## Supported Models

### LLM Models for QA Generation

The tool uses a two-step process for QA generation:

1. **Question Generation**: Uses English prompts to generate context-aware questions in Arabic
2. **Answer Generation**: Uses the generated question and context to create comprehensive answers in Arabic

**Supported Models:**
- `Qwen/Qwen2.5-32B-Instruct` (default)
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-8B`
- Any Hugging Face compatible model

**QA Generation Process:**
- **Step 1**: The LLM analyzes the text context and generates relevant questions in Arabic
- **Step 2**: The LLM uses the generated question and original context to create detailed answers in Arabic
- **Quality Control**: Both questions and answers are cleaned and validated for quality

### Embedding Models

- `Qwen/Qwen2-0.5B-Instruct` (default)
- `Qwen/Qwen3-Embedding-0.6B`
- `sentence-transformers/all-MiniLM-L6-v2`
- Any sentence-transformers compatible model

## Arabic Language Support

The tool includes full support for Arabic text processing and question generation:

- **Two-Step QA Generation**: 
  - **Question Generation**: Uses English prompts for better LLM understanding but generates questions in Arabic
  - **Answer Generation**: Generates comprehensive answers in Arabic based on the question and context
- **Context-Aware Questions**: LLM generates intelligent questions based on the actual content of the text
- **Arabic-Only Output**: All questions and answers are generated in Arabic only
- **Arabic Punctuation**: Proper handling of Arabic punctuation marks
- **Sentence Splitting**: Intelligent sentence boundary detection for Arabic
- **Text Chunking**: Optimal chunking that respects Arabic text structure
- **Unicode Support**: Full Unicode support for Arabic characters

## API Usage

You can also use the tool programmatically:

```python
from config import Settings
from processors import DocumentProcessor, TextChunker, QAGenerator
from utils import EmbeddingGenerator, DataExporter

# Load configuration
settings = Settings("config/default_config.json")

# Initialize processors
document_processor = DocumentProcessor()
chunker = TextChunker()
qa_generator = QAGenerator(offload_model=True)
embedding_generator = EmbeddingGenerator(offload_model=True)

# Process PDF or TXT file
text = document_processor.extract_text("books/sample.pdf")  # or "books/sample.txt"
chunks = chunker.chunk_text(text)
qa_pairs = qa_generator.generate_qa_pairs(chunks)
embeddings = embedding_generator.generate_embeddings(chunks)

# Export results
exporter = DataExporter()
exporter.export_qa_pairs(qa_pairs, "output/qa_pairs.json")
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- PyPDF2
- NumPy
- Pandas

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Hugging Face Transformers
- Uses Qwen models for LLM and embedding generation
- Inspired by Alpaca instruction format

## Supported File Formats

The tool supports the following file formats:

### PDF Files (.pdf)
- **Text Extraction**: Extracts text content from PDF documents
- **Page Limiting**: Use `--max-pages` to limit the number of pages processed
- **Multi-page Support**: Processes all pages by default
- **Image-only Page Handling**: Skips pages with no extractable text

### TXT Files (.txt)
- **Text Extraction**: Reads plain text files directly
- **Encoding Support**: Automatically detects and handles multiple encodings:
  - UTF-8 (with and without BOM)
  - CP1256 (Windows Arabic)
  - ISO-8859-6 (Arabic)
  - Windows-1256 (Arabic)
- **Fallback Handling**: Uses error handling for problematic encodings
- **Paragraph Splitting**: Splits text into paragraphs for processing

### File Size Considerations
- **Large Files**: The tool can handle large documents efficiently
- **Memory Management**: Automatic model offloading prevents memory issues
- **Incremental Processing**: Files are processed in chunks to manage memory usage 