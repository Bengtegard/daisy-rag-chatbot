"""
Simple token management functions for the RAG engine.
"""
import os
import re
from typing import List, Dict, Any, Optional
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from transformers.models.gpt2 import GPT2TokenizerFast
from pydantic import ConfigDict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

# Set this BEFORE importing to avoid the parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TokenHandler:
    """Simple token management"""
    
    def __init__(self, max_model_tokens: int = 32768, reserve_tokens: int = 2048):
        """
        Args:
            max_model_tokens: The model's max token limit
            reserve_tokens: Tokens to reserve for prompt template + response
        """
        self.max_model_tokens = max_model_tokens
        self.max_context_tokens = max_model_tokens - reserve_tokens
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Cache for token counts to avoid re-tokenizing the same content
        self._token_cache = {}
        
        # Improved sentence splitting regex
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching."""
        # Use hash of text as cache key
        cache_key = hash(text)
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = len(self.tokenizer.encode(text))
        return self._token_cache[cache_key]
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Batch tokenize for better performance."""
        # Check cache first
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self._token_cache:
                results.append(self._token_cache[cache_key])
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch process uncached texts
        if uncached_texts:
            encoded_batch = self.tokenizer(uncached_texts, return_tensors=None)
            for i, (text, encoded) in enumerate(zip(uncached_texts, encoded_batch['input_ids'])):
                token_count = len(encoded)
                cache_key = hash(text)
                self._token_cache[cache_key] = token_count
                results[uncached_indices[i]] = token_count
        
        return results
    
    def filter_docs_by_budget(self, documents: List[Document]) -> List[Document]:
        """
        Keep adding docs until we hit token limit.
        Takes highest relevance docs first (assumes they're already sorted).
        """
        if not documents:
            return []
        
        # Batch count tokens for all documents for better performance
        doc_contents = [doc.page_content for doc in documents]
        token_counts = self.count_tokens_batch(doc_contents)
        
        selected_docs = []
        total_tokens = 0
        
        for doc, doc_tokens in zip(documents, token_counts):
            # If this doc fits, add it
            if total_tokens + doc_tokens <= self.max_context_tokens:
                selected_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Try to truncate the last doc if there's meaningful space left
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 150:  # Only if we have decent space
                    truncated_doc = self._truncate_at_sentences(doc, remaining_tokens)
                    if truncated_doc:
                        selected_docs.append(truncated_doc)
                        total_tokens += self.count_tokens(truncated_doc.page_content)
                break  # Stop here regardless
        
        print(f"Selected {len(selected_docs)} docs using {total_tokens} tokens (limit: {self.max_context_tokens})")
        return selected_docs
    
    def _truncate_at_sentences(self, doc: Document, max_tokens: int) -> Optional[Document]:
        """Truncate document at sentence boundaries with improved sentence splitting."""
        # Use regex for better sentence splitting
        sentences = self.sentence_pattern.split(doc.page_content)
        
        # If regex splitting didn't work well, fall back to simple split
        if len(sentences) == 1:
            sentences = doc.page_content.split('. ')
        
        truncated_content = ""
        
        for sentence in sentences:
            # Add back the period if it was removed by splitting
            test_sentence = sentence.strip()
            if not test_sentence:
                continue
                
            if not test_sentence.endswith(('.', '!', '?')):
                test_sentence += "."
            
            test_content = truncated_content + " " + test_sentence if truncated_content else test_sentence
            
            if self.count_tokens(test_content) <= max_tokens:
                truncated_content = test_content
            else:
                break
        
        if len(truncated_content.strip()) < 50:  # Too short to be useful
            return None
        
        # Create new doc with truncated content
        new_doc = Document(
            page_content=truncated_content.strip(),
            metadata={**doc.metadata, 'truncated': True}
        )
        return new_doc
    
    def clear_cache(self):
        """Clear the token count cache if it gets too large."""
        self._token_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get current cache size for monitoring."""
        return len(self._token_cache)

class TokenAwareRetriever(BaseRetriever):
    """Custom retriever that applies token filtering automatically."""

    # Tell Pydantic we’ll add attributes not declared as fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    def __init__(self, base_retriever: BaseRetriever, token_handler: TokenHandler):
        super().__init__()  # Initialize parent class
        self.base_retriever = base_retriever
        self.token_handler = token_handler
        
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
        **kwargs
    ) -> List[Document]:
        docs = self.base_retriever._get_relevant_documents(query, run_manager=run_manager, **kwargs)
        return self.token_handler.filter_docs_by_budget(docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
        **kwargs
    ) -> List[Document]:
        docs = await self.base_retriever._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
        return self.token_handler.filter_docs_by_budget(docs)