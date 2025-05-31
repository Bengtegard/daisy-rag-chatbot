"""
Embedding model handling for the RAG system.
"""

import os
from typing import Dict, Any

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(config: Dict[str, Any]) -> Embeddings:
    """
    Initialize and return an embedding model based on the configuration.
    
    Args:
        config: Dictionary containing embedding configuration
        
    Returns:
        An initialized embedding model
    """
    provider = config.get('provider', 'huggingface').lower()
    
    if provider == 'openai':
        # Check for API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        return OpenAIEmbeddings()
    
    elif provider == 'huggingface':
        model_name = config.get('huggingface_model', 'all-MiniLM-L6-v2')
        return HuggingFaceEmbeddings(model_name=model_name)
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Use 'openai' or 'huggingface'.")