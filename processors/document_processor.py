"""
Document processing module for extracting text from PDF and TXT documents.
"""

import PyPDF2
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF and TXT document processing and text extraction."""
    
    def __init__(self, max_pages: Optional[int] = None):
        self.max_pages = max_pages
        logger.info(f"DocumentProcessor initialized with max_pages={max_pages}")
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from a PDF or TXT file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type based on extension
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            return self._extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt")
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        logger.info(f"Starting text extraction from PDF: {pdf_path}")
        logger.info(f"PDF file size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            with open(pdf_path, 'rb') as file:
                logger.info("Opening PDF file for reading...")
                pdf_reader = PyPDF2.PdfReader(file)
                
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} total pages")
                
                text = ""
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                logger.info(f"Will process {pages_to_process} pages")
                
                for page_num in range(pages_to_process):
                    logger.info(f"Processing page {page_num + 1}/{pages_to_process}")
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        text += page_text + "\n"
                        logger.info(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"Page {page_num + 1}: no text extracted (possibly image-only page)")
                    
                    # Log progress every 10 pages
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Progress: {page_num + 1}/{pages_to_process} pages processed")
                
                final_text_length = len(text.strip())
                logger.info(f"Text extraction completed. Total characters: {final_text_length}")
                logger.info(f"Average characters per page: {final_text_length / pages_to_process:.0f}")
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_text_from_txt(self, txt_path: Path) -> str:
        """Extract text from a TXT file."""
        logger.info(f"Starting text extraction from TXT: {txt_path}")
        logger.info(f"TXT file size: {txt_path.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            # Try different encodings to handle various text files
            encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6', 'windows-1256']
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as file:
                        logger.info(f"Reading TXT file with {encoding} encoding...")
                        text = file.read()
                        
                        if text.strip():
                            final_text_length = len(text.strip())
                            logger.info(f"Text extraction completed. Total characters: {final_text_length}")
                            logger.info(f"Successfully read with {encoding} encoding")
                            return text.strip()
                        else:
                            logger.warning(f"File appears to be empty with {encoding} encoding")
                            
                except UnicodeDecodeError:
                    logger.debug(f"Failed to decode with {encoding} encoding, trying next...")
                    continue
                except Exception as e:
                    logger.error(f"Error reading file with {encoding} encoding: {e}")
                    continue
            
            # If all encodings fail, try with error handling
            logger.warning("All encodings failed, trying with error handling...")
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                final_text_length = len(text.strip())
                logger.info(f"Text extraction completed with error handling. Total characters: {final_text_length}")
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error processing TXT file {txt_path}: {str(e)}")
            raise
    
    def extract_text_by_pages(self, file_path: str) -> List[str]:
        """Extract text from file, returning a list of page texts (for PDFs) or paragraphs (for TXTs)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_text_by_pages_from_pdf(file_path)
        elif file_extension == '.txt':
            return self._extract_text_by_paragraphs_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt")
    
    def _extract_text_by_pages_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF file, returning a list of page texts."""
        logger.info(f"Starting page-by-page text extraction from PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                logger.info("Opening PDF file for reading...")
                pdf_reader = PyPDF2.PdfReader(file)
                
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} total pages")
                
                pages = []
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                logger.info(f"Will process {pages_to_process} pages")
                
                for page_num in range(pages_to_process):
                    logger.info(f"Processing page {page_num + 1}/{pages_to_process}")
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        pages.append(page_text.strip())
                        logger.info(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"Page {page_num + 1}: no text extracted (possibly image-only page)")
                        pages.append("")  # Keep empty string to maintain page order
                    
                    # Log progress every 10 pages
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Progress: {page_num + 1}/{pages_to_process} pages processed")
                
                non_empty_pages = sum(1 for p in pages if p.strip())
                logger.info(f"Page-by-page extraction completed. {non_empty_pages}/{len(pages)} pages contain text")
                
                return pages
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_text_by_paragraphs_from_txt(self, txt_path: Path) -> List[str]:
        """Extract text from TXT file, returning a list of paragraphs."""
        logger.info(f"Starting paragraph-by-paragraph text extraction from TXT: {txt_path}")
        
        try:
            # Read the text file
            text = self._extract_text_from_txt(txt_path)
            
            # Split into paragraphs (double newlines)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            logger.info(f"Paragraph extraction completed. {len(paragraphs)} paragraphs found")
            
            return paragraphs
                
        except Exception as e:
            logger.error(f"Error processing TXT file {txt_path}: {str(e)}")
            raise 