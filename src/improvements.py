"""
Core RAG engine implementation with Groq API and map-reduce.
"""

import os
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.retriever import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from token_limit import TokenHandler


class RAGEngine:
    """RAG (Retrieval-Augmented Generation) engine for the Data Science Assistant."""
    
    def __init__(self, embedding_model: Embeddings, config: Dict[str, Any]):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Model to use for creating embeddings
            config: Configuration dictionary
        """
        self.embedding_model = embedding_model
        self.config = config
        self.vector_db = None
        
        # Initialize LLM with Groq
        llm_config = config.get('llm', {})
        self.llm = ChatGroq(
            model=llm_config.get('model', 'deepseek-r1-distill-llama-70b'),
            temperature=llm_config.get('temperature', 0.6),  # DeepSeek's recommended temperature
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=llm_config.get('max_completion_tokens', 4096),
        )
        
        # Retrieval settings
        retrieval_config = config.get('retrieval', {})
        self.k = retrieval_config.get('k', 6)

        # Initialize token handler with DeepSeek-R1's context window
        self.token_handler = TokenHandler(
            max_model_tokens=32768,  # DeepSeek-R1-Distill context window
            reserve_tokens=2048      
        )
    
    def create_vector_db(self, documents: List[Document], persist_dir: str):
        """
        Create and persist a vector database from documents using FAISS.
        
        Args:
            documents: List of documents to add to the vector store
            persist_dir: Directory to persist the vector store
        """
        
        # Create the FAISS vector store from documents
        self.vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        
        # Save to disk - FAISS uses a directory structure
        self.vector_db.save_local(persist_dir)
        print(f"FAISS vector database successfully created and saved to {persist_dir}")
    
    def load_vector_db(self, persist_dir: str):
        """
        Load a previously created vector database.
        
        Args:
            persist_dir: Directory where the vector store is persisted
        """
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector database directory {persist_dir} not found")
        
        self.vector_db = FAISS.load_local(
            folder_path=persist_dir,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True 
        )
    
    def setup_qa_chain(self):
        """Set up the QA retrieval chain with map-reduce and custom prompts."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Call create_vector_db or load_vector_db first.")
        
        # Map template - extracts relevant info from each document
        map_template = """
        You are analyzing a document excerpt to extract information relevant to a specific question.
        
        Document excerpt:
        {context}
        
        Question: {question}
        
        Extract any information from this document that helps answer the question.
        If this document doesn't contain relevant information, respond with "No relevant information found."
        Be precise and factual - do not make assumptions or add information not present in the document.
        
        Relevant information:
        """
        
        # Reduce template - combines the extracted information
        reduce_template = """
        You are an expert data science assistant specializing in statistics, linear algebra, and programming.
        Based on the extracted information from multiple documents, provide a comprehensive answer to the question.
        
        **CRITICAL:** If none of the extracted information is relevant to the question, or if all extracts say "No relevant information found", respond exactly with:
        "I don't have enough information to answer that question."
        
        **DO NOT HALLUCINATE OR MAKE UP INFORMATION.**
        
        Your answer must be:
        - Based strictly on the provided extracted information
        - Concise and clearly structured
        - Focused on intuitive understanding and practical implications
        - Written as a polished explanation, suitable for academic or professional publication
        
        When appropriate:
        - Define key concepts precisely
        - Include relevant mathematical notation
        - Connect theoretical concepts to practical implementations
        - Include relevant formulas, equations, or matrix operations
        - Use real-world analogies for gaining intuition
        - Provide well-commented, idiomatic code in R, Python and SQL
        - Maintain a formal and authoritative tone throughout
        
        Extracted information from documents:
        {summaries}
        
        Question: {question}
        
        Answer:
        """
        
        # Create prompts
        map_prompt = PromptTemplate(template=map_template, input_variables=["context", "question"])
        reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["summaries", "question"])
        
        # Create chains
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="summaries"
        )

        # Reduce chain
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=None,
        )
        
        # Map-reduce chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",  # Map step input variable
            return_intermediate_steps=False
        )
        
        # Final QA chain
        self.qa_chain = RetrievalQA(
            combine_documents_chain=map_reduce_chain,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": self.k}),
            return_source_documents=True
        )


    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with source documents.
        Uses the qa_chain with map-reduce and sophisticated prompt engineering.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing the answer and source information
        """
        if not hasattr(self, 'qa_chain'):
            self.setup_qa_chain()
        
        # Check if we need token filtering
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.k})
        candidate_docs = retriever.invoke(question)
        
        # Filter by token budget
        filtered_docs = self.token_handler.filter_docs_by_budget(candidate_docs)
                
        # Only modify the chain if we actually filtered out documents
        if len(filtered_docs) < len(candidate_docs):
            
            # Temporarily replace the chain's retriever with filtered documents                         
            class FilteredRetriever(BaseRetriever):
                def __init__(self, docs: List):
                    self.docs = docs
                    
                def _get_relevant_documents(self, query: str) -> List:
                    return self.docs
                    
                async def _aget_relevant_documents(self, query: str) -> List:
                    return self.docs
            
            # Update the chain to use filtered documents
            original_retriever = self.qa_chain.retriever
            self.qa_chain.retriever = FilteredRetriever(filtered_docs)
            result = self.qa_chain.invoke({"query": question})
            # Restore original retriever
            self.qa_chain.retriever = original_retriever
        else:
            # Use the chain directly - no filtering needed
            result = self.qa_chain.invoke({"query": question})
        
        # Build sources from the returned documents
        sources = []
        for doc in result["source_documents"]:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category")
            }
            if doc.metadata.get("page"):
                source_info["page"] = doc.metadata["page"]
            if doc.metadata.get("url"):
                source_info["url"] = doc.metadata["url"]
            
            # Clean None values 
            source_info = {k: v for k, v in source_info.items() if v is not None}
            sources.append(source_info)

        return {
            "answer": result["result"],
            "sources": sources
        }

    def format_sources(self, sources):
        """
        Format sources in alphabetical order with a proper introduction.
        
        Args:
            sources: List of source dictionaries
        
        Returns:
            Formatted string with sources in alphabetical order
        """
        # Sort sources alphabetically by source name
        sorted_sources = sorted(sources, key=lambda x: x['source'])
        
        # Create a list of unique sources with page numbers
        unique_sources = {}
        for source in sorted_sources:
            source_name = source['source'].split('/')[-1]  # Extract filename
            
            if source_name in unique_sources:
                # Add page if not already included
                if source['page'] not in unique_sources[source_name]['pages']:
                    unique_sources[source_name]['pages'].append(source['page'])
            else:
                unique_sources[source_name] = {
                    'pages': [source['page']] if source.get('page') is not None else [],
                    'category': source.get('category', 'unspecified')
                }
        
        # Format the output
        result = "This answer was based on the following sources:\n\n"
        for source_name, info in unique_sources.items():
            if info['pages']:
                pages_str = ", ".join(map(str, sorted(info['pages'])))
                result += f"- {source_name} (pages: {pages_str}) [{info['category']}]\n"
            else:
                result += f"- {source_name} [{info['category']}]\n"
        
        return result