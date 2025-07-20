"""
Content chunking module for splitting HTML into semantic chunks.
"""
import re
from typing import List, Dict, Optional
from .utils import Tokenizer, sanitize_text, load_jsonl, save_jsonl, create_data_dir, logger

class ContentChunker:
    """Chunks content into semantic units of approximately target token size."""
    
    def __init__(self, target_tokens: int = 150, overlap_tokens: int = 20):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = Tokenizer()
    
    def chunk_page(self, page_data: Dict) -> List[Dict]:
        """Chunk a single page into multiple chunks."""
        chunks = []
        
        # Extract different content types
        title = page_data.get('title', '')
        description = page_data.get('description', '')
        text = page_data.get('text', '')
        headings = page_data.get('headings', [])
        
        # Create title chunk if significant
        if title and len(title.strip()) > 10:
            title_chunk = {
                'id': f"{page_data['url']}_title",
                'url': page_data['url'],
                'title': title,
                'content': title,
                'content_type': 'title',
                'tokens': self.tokenizer.count_tokens(title),
                'domain': page_data.get('domain', ''),
                'metadata': {
                    'description': description,
                    'headings': headings
                }
            }
            chunks.append(title_chunk)
        
        # Chunk main text content
        text_chunks = self._chunk_text(text, page_data['url'])
        chunks.extend(text_chunks)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk['page_title'] = title
            chunk['page_description'] = description
            chunk['crawled_at'] = page_data.get('crawled_at', 0)
        
        return chunks
    
    def _chunk_text(self, text: str, url: str) -> List[Dict]:
        """Split text into semantic chunks."""
        if not text.strip():
            return []
        
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.tokenizer.count_tokens(paragraph)
            
            # If paragraph is too long, split it further
            if paragraph_tokens > self.target_tokens:
                # Split by sentences
                sentences = self._split_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = self.tokenizer.count_tokens(sentence)
                    
                    if (self.tokenizer.count_tokens(current_chunk + " " + sentence) <= 
                        self.target_tokens):
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        # Save current chunk
                        if current_chunk.strip():
                            chunks.append(self._create_chunk(
                                current_chunk.strip(), url, chunk_id, 'text'
                            ))
                            chunk_id += 1
                        
                        # Start new chunk with current sentence
                        current_chunk = sentence
            else:
                # Check if adding this paragraph would exceed target
                if (self.tokenizer.count_tokens(current_chunk + " " + paragraph) <= 
                    self.target_tokens):
                    current_chunk += " " + paragraph if current_chunk else paragraph
                else:
                    # Save current chunk
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(
                            current_chunk.strip(), url, chunk_id, 'text'
                        ))
                        chunk_id += 1
                    
                    # Start new chunk with current paragraph
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), url, chunk_id, 'text'
            ))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or HTML paragraph tags
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(self, content: str, url: str, chunk_id: int, content_type: str) -> Dict:
        """Create a chunk dictionary."""
        return {
            'id': f"{url}_chunk_{chunk_id}",
            'url': url,
            'content': content,
            'content_type': content_type,
            'tokens': self.tokenizer.count_tokens(content),
            'chunk_id': chunk_id
        }
    
    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """Chunk multiple pages."""
        all_chunks = []
        
        for page in pages:
            try:
                page_chunks = self.chunk_page(page)
                all_chunks.extend(page_chunks)
                logger.info(f"Created {len(page_chunks)} chunks for {page['url']}")
            except Exception as e:
                logger.error(f"Error chunking page {page['url']}: {e}")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], filename: str = "chunks.jsonl") -> str:
        """Save chunks to JSONL file."""
        data_dir = create_data_dir()
        filepath = f"{data_dir}/{filename}"
        save_jsonl(chunks, filepath)
        logger.info(f"Saved {len(chunks)} chunks to {filepath}")
        return filepath
    
    def load_chunks(self, filename: str = "chunks.jsonl") -> List[Dict]:
        """Load chunks from JSONL file."""
        data_dir = create_data_dir()
        filepath = f"{data_dir}/{filename}"
        return load_jsonl(filepath)

class SemanticChunker(ContentChunker):
    """Advanced chunker that respects semantic boundaries."""
    
    def __init__(self, target_tokens: int = 150, overlap_tokens: int = 20):
        super().__init__(target_tokens, overlap_tokens)
        self.semantic_boundaries = [
            r'\n\s*\n',  # Paragraph breaks
            r'[.!?]\s+',  # Sentence endings
            r'[:;]\s+',   # Colons and semicolons
            r'\s+and\s+', # Conjunctions
            r'\s+or\s+',
            r'\s+but\s+',
        ]
    
    def _chunk_text(self, text: str, url: str) -> List[Dict]:
        """Split text using semantic boundaries."""
        if not text.strip():
            return []
        
        # Find all potential split points
        split_points = self._find_split_points(text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for segment in split_points:
            segment_tokens = self.tokenizer.count_tokens(segment)
            
            if segment_tokens > self.target_tokens:
                # Recursively split large segments
                sub_chunks = self._chunk_large_segment(segment, url, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                if (self.tokenizer.count_tokens(current_chunk + " " + segment) <= 
                    self.target_tokens):
                    current_chunk += " " + segment if current_chunk else segment
                else:
                    if current_chunk.strip():
                        chunks.append(self._create_chunk(
                            current_chunk.strip(), url, chunk_id, 'text'
                        ))
                        chunk_id += 1
                    current_chunk = segment
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), url, chunk_id, 'text'
            ))
        
        return chunks
    
    def _find_split_points(self, text: str) -> List[str]:
        """Find natural split points in text."""
        # Start with the text as one segment
        segments = [text]
        
        # Apply each boundary pattern
        for pattern in self.semantic_boundaries:
            new_segments = []
            for segment in segments:
                if self.tokenizer.count_tokens(segment) <= self.target_tokens:
                    new_segments.append(segment)
                else:
                    # Split at this boundary
                    parts = re.split(pattern, segment)
                    new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        return segments
    
    def _chunk_large_segment(self, segment: str, url: str, base_id: int) -> List[Dict]:
        """Handle segments that are still too large."""
        # Simple character-based splitting for very large segments
        words = segment.split()
        chunks = []
        current_chunk = ""
        chunk_id = base_id
        
        for word in words:
            if (self.tokenizer.count_tokens(current_chunk + " " + word) <= 
                self.target_tokens):
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(), url, chunk_id, 'text'
                    ))
                    chunk_id += 1
                current_chunk = word
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), url, chunk_id, 'text'
            ))
        
        return chunks

def chunk_crawled_pages(input_file: str = "crawled_pages.jsonl", 
                       output_file: str = "chunks.jsonl",
                       use_semantic: bool = True) -> List[Dict]:
    """Convenience function to chunk crawled pages."""
    from .utils import load_jsonl
    
    data_dir = create_data_dir()
    input_path = f"{data_dir}/{input_file}"
    
    pages = load_jsonl(input_path)
    if not pages:
        logger.warning(f"No pages found in {input_path}")
        return []
    
    chunker = SemanticChunker() if use_semantic else ContentChunker()
    chunks = chunker.chunk_pages(pages)
    
    output_path = chunker.save_chunks(chunks, output_file)
    return chunks 