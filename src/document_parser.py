"""
Document parsing utilities for multiple file formats
"""
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse documents in various formats and extract metadata"""
    
    def __init__(self):
        self.supported_formats = {
            '.md': self._parse_markdown,
            '.txt': self._parse_text,
            '.docx': self._parse_docx,
            '.pdf': self._parse_pdf
        }
    
    def parse_document(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a document and return content + metadata
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (content, metadata)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Parse content
        content, extracted_metadata = self.supported_formats[extension](file_path)
        
        # Build complete metadata
        metadata = self._build_metadata(file_path, extracted_metadata)
        
        return content, metadata
    
    def _parse_markdown(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse Markdown file with frontmatter support"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Extract YAML frontmatter if present
        frontmatter = {}
        content = raw_content
        
        if raw_content.startswith('---\n'):
            parts = raw_content.split('---\n', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    content = parts[2]
                    
                    # Convert date objects to ISO strings for JSON serialization
                    for key, value in frontmatter.items():
                        if hasattr(value, 'isoformat'):  # datetime/date objects
                            frontmatter[key] = value.isoformat()
                            
                except yaml.YAMLError:
                    logger.warning(f"Invalid YAML frontmatter in {file_path}")
        
        # Convert markdown to text for processing
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text_content = soup.get_text()
        
        return text_content, frontmatter
    
    def _parse_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, {}
    
    def _parse_docx(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse DOCX file"""
        if Document is None:
            raise ImportError("python-docx is required for DOCX support")
        
        doc = Document(file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)
        
        # Extract basic metadata from document properties
        metadata = {}
        if hasattr(doc.core_properties, 'title') and doc.core_properties.title:
            metadata['title'] = doc.core_properties.title
        if hasattr(doc.core_properties, 'author') and doc.core_properties.author:
            metadata['author'] = doc.core_properties.author
        if hasattr(doc.core_properties, 'created') and doc.core_properties.created:
            metadata['created'] = doc.core_properties.created.isoformat()
        
        return '\n'.join(content), metadata
    
    def _parse_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF file"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF support")
        
        content = []
        metadata = {}
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Extract text from all pages
            for page in reader.pages:
                content.append(page.extract_text())
            
            # Extract PDF metadata
            if reader.metadata:
                pdf_meta = reader.metadata
                if '/Title' in pdf_meta:
                    metadata['title'] = pdf_meta['/Title']
                if '/Author' in pdf_meta:
                    metadata['author'] = pdf_meta['/Author']
                if '/CreationDate' in pdf_meta:
                    metadata['created'] = str(pdf_meta['/CreationDate'])
        
        return '\n'.join(content), metadata
    
    def _build_metadata(self, file_path: Path, extracted_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete metadata for a document"""
        stat = file_path.stat()
        
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
        }
        
        # Add extracted metadata
        metadata.update(extracted_metadata)
        
        # Extract topics from file path and content metadata
        topics = self._extract_topics(file_path, extracted_metadata)
        if topics:
            metadata['topics'] = topics
        
        # Set priority based on metadata or defaults
        metadata['priority'] = self._determine_priority(extracted_metadata)
        
        # Extract or generate title
        if 'title' not in metadata:
            metadata['title'] = self._extract_title(file_path.stem)
        
        return metadata
    
    def _extract_topics(self, file_path: Path, metadata: Dict[str, Any]) -> List[str]:
        """Extract topics from file path and metadata"""
        topics = []
        
        # From metadata
        if 'topics' in metadata:
            if isinstance(metadata['topics'], list):
                topics.extend(metadata['topics'])
            elif isinstance(metadata['topics'], str):
                topics.append(metadata['topics'])
        
        # From file path (directory names can indicate topics)
        path_parts = file_path.parts[:-1]  # Exclude filename
        for part in path_parts:
            if part not in ['documents', 'data', 'src']:  # Skip common dirs
                topics.append(part.replace('_', ' ').replace('-', ' '))
        
        # From tags in metadata
        if 'tags' in metadata:
            if isinstance(metadata['tags'], list):
                topics.extend(metadata['tags'])
            elif isinstance(metadata['tags'], str):
                topics.extend(metadata['tags'].split(','))
        
        return list(set(topics))  # Remove duplicates
    
    def _determine_priority(self, metadata: Dict[str, Any]) -> str:
        """Determine document priority based on metadata"""
        if 'priority' in metadata:
            priority = str(metadata['priority']).lower()
            if priority in ['high', 'medium', 'low']:
                return priority
        
        # Default priority based on other factors
        if 'important' in str(metadata).lower() or 'urgent' in str(metadata).lower():
            return 'high'
        
        return 'medium'
    
    def _extract_title(self, filename: str) -> str:
        """Extract a readable title from filename"""
        # Remove common prefixes/suffixes
        title = filename
        title = re.sub(r'^\d{4}-\d{2}-\d{2}[-_]?', '', title)  # Remove date prefix
        title = re.sub(r'[-_]?v?\d+$', '', title)  # Remove version suffix
        
        # Replace separators with spaces and title case
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title


def chunk_text(text: str, chunk_size: int = 512, overlap_size: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap_size: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', max(start, end - 100), end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap_size)
        
        if start >= len(text):
            break
    
    return chunks
