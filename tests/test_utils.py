"""
Unit tests for utils module.
"""
import pytest
import os
import tempfile
import json
from src.utils import (
    Tokenizer, sanitize_text, extract_domain, is_valid_url,
    get_env_var, create_data_dir, save_jsonl, load_jsonl,
    chunk_list, safe_request
)

class TestTokenizer:
    """Test Tokenizer class."""
    
    def test_count_tokens(self):
        """Test token counting."""
        tokenizer = Tokenizer()
        text = "Hello world, this is a test."
        count = tokenizer.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)
    
    def test_truncate_to_tokens(self):
        """Test token truncation."""
        tokenizer = Tokenizer()
        text = "This is a very long text that should be truncated"
        max_tokens = 5
        truncated = tokenizer.truncate_to_tokens(text, max_tokens)
        assert tokenizer.count_tokens(truncated) <= max_tokens

class TestTextProcessing:
    """Test text processing functions."""
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        dirty_text = "  This   has   extra   spaces  &nbsp; and &amp; entities  "
        clean_text = sanitize_text(dirty_text)
        assert "  " not in clean_text  # No double spaces
        assert "&nbsp;" not in clean_text
        assert "&amp;" not in clean_text
        assert clean_text.strip() == clean_text
    
    def test_extract_domain(self):
        """Test domain extraction."""
        url = "https://www.example.com/path/to/page?param=value"
        domain = extract_domain(url)
        assert domain == "www.example.com"
    
    def test_is_valid_url(self):
        """Test URL validation."""
        assert is_valid_url("https://example.com")
        assert is_valid_url("http://example.com/path")
        assert not is_valid_url("not-a-url")
        assert not is_valid_url("")

class TestFileOperations:
    """Test file operation functions."""
    
    def test_create_data_dir(self):
        """Test data directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            data_dir = create_data_dir()
            assert os.path.exists(data_dir)
            assert os.path.isdir(data_dir)
            
            os.chdir(original_cwd)
    
    def test_save_and_load_jsonl(self):
        """Test JSONL save and load operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = [
                {"id": 1, "text": "Hello"},
                {"id": 2, "text": "World"}
            ]
            
            filepath = os.path.join(temp_dir, "test.jsonl")
            save_jsonl(test_data, filepath)
            
            assert os.path.exists(filepath)
            
            loaded_data = load_jsonl(filepath)
            assert len(loaded_data) == 2
            assert loaded_data[0]["id"] == 1
            assert loaded_data[1]["text"] == "World"

class TestUtilityFunctions:
    """Test other utility functions."""
    
    def test_chunk_list(self):
        """Test list chunking."""
        test_list = [1, 2, 3, 4, 5, 6]
        chunks = chunk_list(test_list, 2)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [3, 4]
        assert chunks[2] == [5, 6]
    
    def test_get_env_var(self):
        """Test environment variable retrieval."""
        # Test with default value
        value = get_env_var("NONEXISTENT_VAR", "default")
        assert value == "default"
        
        # Test with None default
        value = get_env_var("NONEXISTENT_VAR")
        assert value is None

class TestRetryDecorator:
    """Test retry functionality."""
    
    def test_retry_on_failure(self):
        """Test retry decorator."""
        from src.utils import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3 