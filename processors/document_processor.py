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
                
                text_parts = []
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                logger.info(f"Will process {pages_to_process} pages")
                
                for page_num in range(pages_to_process):
                    logger.info(f"Processing page {page_num + 1}/{pages_to_process}")
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        # Clean and format the page text
                        cleaned_page_text = self._clean_page_text(page_text, page_num + 1)
                        text_parts.append(cleaned_page_text)
                        logger.info(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"Page {page_num + 1}: no text extracted (possibly image-only page)")
                    
                    # Log progress every 10 pages
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Progress: {page_num + 1}/{pages_to_process} pages processed")
                
                # Join all text parts with proper delimiters
                final_text = self._join_text_parts(text_parts)
                final_text_length = len(final_text.strip())
                logger.info(f"Text extraction completed. Total characters: {final_text_length}")
                logger.info(f"Average characters per page: {final_text_length / pages_to_process:.0f}")
                
                return final_text.strip()
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _clean_page_text(self, page_text: str, page_num: int) -> str:
        """Clean and format page text with proper delimiters."""
        # Remove excessive whitespace
        page_text = ' '.join(page_text.split())
        
        # Add page header
        page_header = f"\n\n=== PAGE {page_num} ===\n\n"
        
        # Split into paragraphs and add proper spacing
        paragraphs = [p.strip() for p in page_text.split('\n') if p.strip()]
        formatted_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            # Add paragraph separator
            if i > 0:
                formatted_paragraphs.append("\n\n")
            formatted_paragraphs.append(paragraph)
        
        # Join paragraphs and add page header
        cleaned_text = page_header + ''.join(formatted_paragraphs)
        
        return cleaned_text
    
    def _join_text_parts(self, text_parts: List[str]) -> str:
        """Join text parts with proper delimiters."""
        if not text_parts:
            return ""
        
        # Join with section separators
        joined_text = "\n\n" + "=" * 50 + "\n\n".join(text_parts)
        
        # Add final section separator
        joined_text += "\n\n" + "=" * 50 + "\n\n"
        
        return joined_text
    
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
                            # Clean and format the text
                            cleaned_text = self._clean_txt_text(text)
                            final_text_length = len(cleaned_text.strip())
                            logger.info(f"Text extraction completed. Total characters: {final_text_length}")
                            logger.info(f"Successfully read with {encoding} encoding")
                            return cleaned_text.strip()
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
                cleaned_text = self._clean_txt_text(text)
                final_text_length = len(cleaned_text.strip())
                logger.info(f"Text extraction completed with error handling. Total characters: {final_text_length}")
                return cleaned_text.strip()
                
        except Exception as e:
            logger.error(f"Error processing TXT file {txt_path}: {str(e)}")
            raise
    
    def _clean_txt_text(self, text: str) -> str:
        """Clean and format TXT text with proper delimiters."""
        # Remove excessive whitespace but preserve paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Clean the line but preserve spacing
                cleaned_line = ' '.join(line.split())
                cleaned_lines.append(cleaned_line)
            else:
                # Preserve empty lines as paragraph separators
                cleaned_lines.append("")
        
        # Join lines with proper spacing
        formatted_text = "\n".join(cleaned_lines)
        
        # Add section separators
        section_header = "\n\n" + "=" * 50 + "\nTEXT DOCUMENT\n" + "=" * 50 + "\n\n"
        section_footer = "\n\n" + "=" * 50 + "\n\n"
        
        return section_header + formatted_text + section_footer
    
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