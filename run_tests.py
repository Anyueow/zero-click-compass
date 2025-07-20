#!/usr/bin/env python3
"""
Simple test runner for zero-click-compass.
"""
import subprocess
import sys
import os

def run_tests():
    """Run the test suite."""
    print("🧪 Running Zero-Click Compass Tests")
    print("=" * 50)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
    
    # Run tests
    test_dir = "tests"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory '{test_dir}' not found")
        return False
    
    print(f"📁 Running tests from {test_dir}")
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_dir, 
            "-v", 
            "--tb=short"
        ], capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_smoke_test():
    """Run a quick smoke test of the pipeline."""
    print("\n🔥 Running Smoke Test")
    print("=" * 30)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.utils import Tokenizer, sanitize_text
        from src.chunk import ContentChunker
        from src.expand import QueryExpander
        print("✅ All imports successful")
        
        # Test basic functionality
        print("🔧 Testing basic functionality...")
        
        # Test tokenizer
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("Hello world")
        assert count > 0
        print("✅ Tokenizer working")
        
        # Test text sanitization
        clean_text = sanitize_text("  Test   text  &nbsp;  ")
        assert "  " not in clean_text
        print("✅ Text sanitization working")
        
        # Test chunker
        chunker = ContentChunker(target_tokens=100)
        test_page = {
            'url': 'https://example.com',
            'title': 'Test Page',
            'text': 'This is a test page with some content.',
            'domain': 'example.com'
        }
        chunks = chunker.chunk_page(test_page)
        assert len(chunks) > 0
        print("✅ Chunker working")
        
        print("✅ Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧭 Zero-Click Compass Test Suite")
    print("=" * 40)
    
    # Run smoke test first
    smoke_success = run_smoke_test()
    
    # Run full test suite
    test_success = run_tests()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    print(f"Smoke Test: {'✅ PASSED' if smoke_success else '❌ FAILED'}")
    print(f"Full Tests: {'✅ PASSED' if test_success else '❌ FAILED'}")
    
    if smoke_success and test_success:
        print("\n🎉 All tests passed! Zero-Click Compass is ready to use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        sys.exit(1) 