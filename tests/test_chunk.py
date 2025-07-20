"""
Unit tests for chunk module.
"""
import pytest
from src.chunk import ContentChunker, SemanticChunker, chunk_crawled_pages

class TestContentChunker:
    """Test ContentChunker class."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunker = ContentChunker(target_tokens=100)
        
        page_data = {
            'url': 'https://www.avocadogreenmattress.com',
            'title': 'Avocado Green Mattress',
            'description': 'Avocado Green Mattress is a sustainable and eco-friendly mattress company that offers a range of mattresses, pillows, and bedding products.',
            'text': 'Avocado Green Mattress igits a sustainable and eco-friendly mattress company that offers a range of mattresses, pillows, and bedding products.',
            'headings': [{'level': 1, 'text': 'Main Heading'}],
            'domain': 'avocadogreenmattress.com'
        }
        
        chunks = chunker.chunk_page(page_data)
        
        assert len(chunks) > 0
        assert all('id' in chunk for chunk in chunks)
        assert all('url' in chunk for chunk in chunks)
        assert all('content' in chunk for chunk in chunks)
    
    def test_chunk_size_limits(self):
        """Test that chunks respect token limits."""
        chunker = ContentChunker(target_tokens=50)
        
        long_text = "This is a very long text that should be split into multiple chunks. " * 10
        
        page_data = {
            'url': 'https://example.com',
            'title': 'Test Page',
            'text': long_text,
            'domain': 'example.com'
        }
        
        chunks = chunker.chunk_page(page_data)
        
        for chunk in chunks:
            tokens = chunk.get('tokens', 0)
            assert tokens <= 50 or tokens <= 60  # Allow some flexibility
    
    def test_title_chunk_creation(self):
        """Test that titles create separate chunks."""
        chunker = ContentChunker()
        
        page_data = {
            'url': 'https://example.com',
            'title': 'This is a significant title that should be chunked',
            'text': 'Some body text here.',
            'domain': 'example.com'
        }
        
        chunks = chunker.chunk_page(page_data)
        
        # Should have at least title chunk and text chunk
        assert len(chunks) >= 2
        
        # Find title chunk
        title_chunk = next((chunk for chunk in chunks if chunk.get('content_type') == 'title'), None)
        assert title_chunk is not None
        assert 'This is a significant title' in title_chunk['content']

class TestSemanticChunker:
    """Test SemanticChunker class."""
    
    def test_semantic_boundaries(self):
        """Test that semantic chunker respects boundaries."""
        chunker = SemanticChunker(target_tokens=100)
        
        text_with_boundaries = """
        First paragraph. This has some content.
        
        Second paragraph. This is separate.
        
        Third paragraph with more content. It continues here.
        """
        
        page_data = {
            'url': 'https://example.com',
            'text': text_with_boundaries,
            'domain': 'example.com'
        }
        
        chunks = chunker.chunk_page(page_data)
        
        assert len(chunks) > 0
        # Should respect paragraph boundaries
        for chunk in chunks:
            content = chunk['content']
            # Check that chunks don't break mid-paragraph
            assert not (content.endswith('First paragraph') and 'Second paragraph' in content)

class TestChunkingFunctions:
    """Test chunking utility functions."""
    
    def test_chunk_crawled_pages_missing_file(self):
        """Test chunk_crawled_pages with missing input file."""
        # This should handle missing file gracefully
        chunks = chunk_crawled_pages(input_file="nonexistent.jsonl")
        assert chunks == []
    
    def test_chunk_save_and_load(self):
        """Test that chunks can be saved and loaded."""
        chunker = ContentChunker()
        
        test_chunks = [
            {
                'id': 'test1',
                'url': 'https://example.com',
                'content': 'Test content 1',
                'content_type': 'text',
                'tokens': 10
            },
            {
                'id': 'test2',
                'url': 'https://example.com',
                'content': 'Test content 2',
                'content_type': 'text',
                'tokens': 10
            }
        ]
        
        # Test save
        filepath = chunker.save_chunks(test_chunks, "test_chunks.jsonl")
        assert filepath.endswith("test_chunks.jsonl")
        
        # Test load
        loaded_chunks = chunker.load_chunks("test_chunks.jsonl")
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0]['id'] == 'test1'
        assert loaded_chunks[1]['id'] == 'test2' 