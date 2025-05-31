# Standard library imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import re
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Local application imports
from src.rag_engine import RAGEngine
from src.token_manager import TokenHandler
from src.math_render import StreamlitMathRenderer

# Load environment variables
load_dotenv()

# Hidden setup values
db_path = "./vector_db"
api_key_status = "Set" if os.getenv("GROQ_API_KEY") else "Not Set"

if not os.getenv("GROQ_API_KEY"):
    st.warning("Please set your GROQ_API_KEY environment variable")

# Page config
st.set_page_config(
    page_title="Daisy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-family: monospace;
        font-weight: bold;
        color: #D36A3F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #2B2A29;
        font-family: monospace;
        color: #1B9E77;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #FFFFFA;
        font-family: monospace;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sources-box {
        background-color: #f8f9fa;
        font-family: monospace;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .stButton > button {
        width: 100%;
        font-family:monospace;
        background-color: #2a3545;
        color: white;
        border-radius: 5px;
    }
    .math-container {
        background-color: #f8f9fa;
        font-family:monospace;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #FF7A00;
    }
    textarea {
        background-color: #1F2937 !important;
        color: #FFFFFA !important;
        font-family: monospace !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load configuration for the RAG engine."""
    return {
        'llm': {
            'model': 'deepseek-r1-distill-llama-70b',
            'temperature': 0.6,
            'max_completion_tokens': 4096
        },
        'retrieval': {
            'k': 8
        }
    }

@st.cache_resource
def initialize_rag_engine():
    """Initialize the RAG engine with caching."""
    try:
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Load config
        config = load_config()
        
        # Initialize RAG engine
        rag_engine = RAGEngine(embedding_model, config)
        
        return rag_engine, None
    except Exception as e:
        return None, str(e)

def safe_math_renderer(text: str) -> None:
    """
    Safe math renderer with error handling.
    Falls back to regular markdown if math rendering fails.
    """
    if not text or not isinstance(text, str):
        st.markdown("*No content to display*")
        return
    
    try:
        renderer = StreamlitMathRenderer()
        renderer.smart_render(text)
    except Exception as e:
        st.error(f"Math rendering error: {str(e)}")
        # Fallback to regular markdown
        st.markdown(text)

def display_answer_with_math(answer_text):
    """
    Display answer with proper math rendering and enhanced formatting.
    """
    if not answer_text:
        st.warning("No answer received.")
        return
    
    #st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("**Answer:**")
    
    # Use the safe math renderer for the answer
    safe_math_renderer(answer_text)
    
def format_sources_safely(sources):
    """Safely format sources with error handling."""
    if not sources:
        return "No sources available."
    
    try:
        formatted = []
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                # Extract relevant information safely
                source_info = f"**Source {i}:**"
                if 'metadata' in source and isinstance(source['metadata'], dict):
                    metadata = source['metadata']
                    if 'source' in metadata:
                        source_info += f" {metadata['source']}"
                    if 'page' in metadata:
                        source_info += f" (Page {metadata['page']})"
                
                formatted.append(source_info)
                
                # Add content if available
                if 'page_content' in source:
                    content = str(source['page_content'])[:200] + "..." if len(str(source['page_content'])) > 200 else str(source['page_content'])
                    formatted.append(f"*{content}*")
            else:
                formatted.append(f"**Source {i}:** {str(source)[:100]}...")
        
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error formatting sources: {str(e)}"

def main():
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Header
    st.markdown('<h1 class="main-header"> Daisy</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background-color: #D36A3F;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    ">
    <h4>🌼 Meet Daisy – A Smart Study Companion</h4>
    <p><strong>Daisy</strong> was created to help bring clarity and structure to my learning in <strong>data science, statistics, R, and Python</strong>.<br>
    Powered by a <strong>RAG model</strong> and built from curated materials—including <em>personal bookmarks</em> and <em>school literature</em>—Daisy generates answers on key topics and provides them with <em>sources</em> for more reliable knowledge.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG engine
    with st.spinner("Initializing RAG engine..."):
        rag_engine, error = initialize_rag_engine()
    
    if error:
        st.error(f"Failed to initialize RAG engine: {error}")
        st.stop()
    
    # Load vector database using RAG engine's method
    try:
        with st.spinner("Loading vector database..."):
            rag_engine.load_vector_db(db_path)  
    except FileNotFoundError as e:
        st.error(f"Vector database not found: {str(e)}")
        st.info("Please ensure your vector database is created and the path is correct.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        st.stop()
    
    # Main interface
    st.markdown("---")
    
    # Question input
    question = st.text_area(
        "Ask your question:",
        height=100,
        placeholder="e.g., What is a neural network?",
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("Get Answer", type="primary")
    
    # Process question
    if ask_button and question and question.strip():
        with st.spinner("Thinking... This may take a moment with map-reduce processing."):
            try:
                # Get answer from RAG engine
                result = rag_engine.ask(question.strip())
                
                if not result:
                    st.error("No result received from RAG engine.")
                    return
                
                # Display question
                #st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.markdown(f"**Question:** {question}")
                st.markdown("---")          

                # Display answer with math rendering
                answer = result.get("answer", "No answer provided.")
                display_answer_with_math(answer)
                
                # Display sources
                sources = result.get("sources", [])
                if sources:
                    st.markdown('<div class="sources-box">', unsafe_allow_html=True)
                    st.markdown("**Sources:**")
                    
                    # Format sources safely
                    try:
                        if hasattr(rag_engine, 'format_sources'):
                            formatted_sources = rag_engine.format_sources(sources)
                        else:
                            formatted_sources = format_sources_safely(sources)
                        st.markdown(formatted_sources)
                    except Exception as e:
                        st.warning(f"Error formatting sources: {str(e)}")
                        st.write("Raw sources:", sources[:3])  # Show first 3 sources
                    
                    st.markdown("---")
                    
                    # Show raw source details in expander   
                    with st.expander(" Detailed Source Information"):
                        for i, source in enumerate(sources[:5], 1):  # Limit to first 5 sources
                            st.write(f"**Source {i}:**")
                            if isinstance(source, dict):
                                st.json(source)
                            else:
                                st.write(str(source))
                
                # Add to conversation history
                st.session_state.conversation_history.append((question, answer))
                
                # Limit conversation history to last 10 items
                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                
                # Show helpful error messages
                error_str = str(e).lower()
                if "413" in str(e) or "request too large" in error_str:
                    st.warning("Token limit exceeded. Try asking a shorter question or reduce context.")
                elif "rate_limit" in error_str:
                    st.warning("Rate limit exceeded. Please wait a moment before asking another question.")
                elif "timeout" in error_str:
                    st.warning("Request timed out. Please try again.")
                elif "connection" in error_str:
                    st.warning("Connection error. Please check your internet connection and try again.")
                else:
                    st.info("Please try rephrasing your question or try again in a moment.")
    
    elif ask_button and (not question or not question.strip()):
        st.warning("Please enter a question!")
    
    # Footer with example questions
    st.markdown("---")
    st.markdown("### Example Questions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Statistics & Theory:**
        - What is the Central Limit Theorem?
        - Explain the difference between Type I and Type II errors
        - How does regularization work in linear regression?
        """)
    
    with col2:
        st.markdown("""
        **Programming & Implementation:**
        - Show me how to create a confusion matrix in Python
        - What's the difference between pandas and NumPy?
        -  How do I implement a simple neural network in scikit-learn?
        """)
    
    # Show conversation history in sidebar
    if st.session_state.conversation_history:
        with st.sidebar:
            st.markdown("---")
            st.header("    Recent Questions")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                if 'selected_question' in st.session_state:
                    del st.session_state.selected_question
                if 'selected_answer' in st.session_state:
                    del st.session_state.selected_answer
                st.rerun()
            
            for i, (q, a) in enumerate(reversed(st.session_state.conversation_history[-5:])):
                # Truncate long questions for display
                display_q = q[:30] + "..." if len(q) > 30 else q
                if st.button(f"Q{len(st.session_state.conversation_history)-i}: {display_q}", key=f"hist_{i}"):
                    st.session_state.selected_question = q
                    st.session_state.selected_answer = a 
                    st.rerun()
    
    # Display selected historical Q&A if any
    if "selected_question" in st.session_state and "selected_answer" in st.session_state:
        st.markdown("---")
        st.markdown("### Previous Question & Answer")
        
        # Add a button to clear the selection
        if st.button("Close Previous Q&A"):
            if 'selected_question' in st.session_state:
                del st.session_state.selected_question
            if 'selected_answer' in st.session_state:
                del st.session_state.selected_answer
            st.rerun()
        
        st.markdown(f"**Q:** {st.session_state.selected_question}")
        st.markdown("**A:**")
        safe_math_renderer(st.session_state.selected_answer)

if __name__ == "__main__":
    main()