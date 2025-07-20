"""
Embedding module for generating embeddings and building FAISS index.
"""
import os
import time
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai

from .utils import (
    get_env_var, create_data_dir, load_jsonl, save_jsonl, 
    retry_on_failure, logger, chunk_list
)

class EmbeddingGenerator:
    """Generate embeddings using Google Gemini API."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
        
        # For embeddings, we'll use the embedding model
        self.embedding_model = genai.get_model('embedding-001')
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text."""
        try:
            # Truncate text if too long (Gemini has limits)
            if len(text) > 8000:
                text = text[:8000]
            
            # Use the embed_content function directly from genai
            import google.generativeai as genai
            result = genai.embed_content(
                model='models/embedding-001',
                content=text
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[Optional[List[float]]]:
        """Get embeddings for a batch of texts."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to chunks."""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk['embedding'] = embedding
            else:
                logger.warning(f"Failed to get embedding for chunk {chunk.get('id', 'unknown')}")
        
        return chunks

class FAISSIndex:
    """FAISS index for efficient similarity search."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.data_dir = create_data_dir()
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build FAISS index from chunks with embeddings."""
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Filter chunks with embeddings
        valid_chunks = [chunk for chunk in chunks if 'embedding' in chunk]
        if not valid_chunks:
            raise ValueError("No chunks with embeddings found")
        
        self.chunks = valid_chunks
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in valid_chunks], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        logger.info(f"Built FAISS index with {len(valid_chunks)} chunks")
    
    def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query embedding
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        # Return (chunk_index, score) pairs
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i != -1:  # FAISS returns -1 for invalid indices
                results.append((i, float(score)))
        
        return results
    
    def get_chunk_by_index(self, index: int) -> Optional[Dict]:
        """Get chunk by index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None
    
    def search_similar(self, query: str, k: int = 10) -> List[Dict]:
        """Search for chunks similar to a query string."""
        # Get query embedding
        embedding_generator = EmbeddingGenerator()
        query_embedding = embedding_generator.get_embedding(query)
        if not query_embedding:
            logger.error("Failed to get query embedding")
            return []
        
        # Search using the search method
        results = self.search(query_embedding, k)
        
        # Return chunks with scores
        similar_chunks = []
        for index, score in results:
            chunk = self.get_chunk_by_index(index)
            if chunk:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score
                similar_chunks.append(chunk_copy)
        
        return similar_chunks
    
    def save_index(self, filename: str = "faiss_index") -> str:
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        index_path = os.path.join(self.data_dir, f"{filename}.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save chunks metadata
        chunks_path = os.path.join(self.data_dir, f"{filename}_chunks.jsonl")
        save_jsonl(self.chunks, chunks_path)
        
        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}")
        return index_path
    
    def load_index(self, filename: str = "faiss_index") -> None:
        """Load FAISS index and chunks from disk."""
        # Load FAISS index
        index_path = os.path.join(self.data_dir, f"{filename}.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks metadata
        chunks_path = os.path.join(self.data_dir, f"{filename}_chunks.jsonl")
        self.chunks = load_jsonl(chunks_path)
        
        logger.info(f"Loaded index with {len(self.chunks)} chunks")

class EmbeddingPipeline:
    """Complete pipeline for embedding chunks and building search index."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.embedding_generator = EmbeddingGenerator(model)
        self.faiss_index = FAISSIndex()
        self.data_dir = create_data_dir()
    
    def process_chunks(self, chunks: List[Dict], save_intermediate: bool = True) -> FAISSIndex:
        """Process chunks through the complete embedding pipeline."""
        logger.info(f"Starting embedding pipeline for {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunks_with_embeddings = self.embedding_generator.embed_chunks(chunks)
        
        # Save intermediate results
        if save_intermediate:
            embedded_chunks_path = os.path.join(self.data_dir, "embedded_chunks.jsonl")
            save_jsonl(chunks_with_embeddings, embedded_chunks_path)
            logger.info(f"Saved embedded chunks to {embedded_chunks_path}")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.faiss_index.build_index(chunks_with_embeddings)
        
        # Save index
        self.faiss_index.save_index()
        
        logger.info("Embedding pipeline completed successfully")
        return self.faiss_index
    
    def search_similar(self, query: str, k: int = 10) -> List[Dict]:
        """Search for chunks similar to a query."""
        # Get query embedding
        query_embedding = self.embedding_generator.get_embedding(query)
        if not query_embedding:
            logger.error("Failed to get query embedding")
            return []
        
        # Search index
        results = self.faiss_index.search(query_embedding, k)
        
        # Return chunks with scores
        similar_chunks = []
        for index, score in results:
            chunk = self.faiss_index.get_chunk_by_index(index)
            if chunk:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score
                similar_chunks.append(chunk_copy)
        
        return similar_chunks

def embed_and_index(input_file: str = "chunks.jsonl", 
                   output_prefix: str = "faiss_index") -> FAISSIndex:
    """Convenience function to embed chunks and build index."""
    # Load chunks
    data_dir = create_data_dir()
    chunks_path = os.path.join(data_dir, input_file)
    chunks = load_jsonl(chunks_path)
    
    if not chunks:
        logger.warning(f"No chunks found in {chunks_path}")
        return None
    
    # Process through pipeline
    pipeline = EmbeddingPipeline()
    return pipeline.process_chunks(chunks)

def load_existing_index(index_name: str = "faiss_index") -> FAISSIndex:
    """Load an existing FAISS index."""
    faiss_index = FAISSIndex()
    faiss_index.load_index(index_name)
    return faiss_index 