"""
PDF processing module for extracting text from PDF documents.
"""

import PyPDF2
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing and text extraction."""
    
    def __init__(self, max_pages: Optional[int] = None):
        self.max_pages = max_pages
        logger.info(f"PDFProcessor initialized with max_pages={max_pages}")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
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
    
    def extract_text_by_pages(self, pdf_path: str) -> List[str]:
        """Extract text from PDF file, returning a list of page texts."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
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