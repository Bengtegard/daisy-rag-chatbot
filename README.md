Data Science RAG Assistant

A personalized Retrieval-Augmented Generation (RAG) chatbot for data science learning materials.
Overview

This project creates a chatbot that can answer questions based on your personal collection of data science learning materials, including:

    Statistics textbooks and references
    Python and R programming guides
    Machine learning documentation
    Personal notes and code examples

Features

    Process multiple document types (PDF, TXT, CSV, Jupyter notebooks)
    Index web pages with valuable data science content
    Provide answers with references to specific sources and page numbers
    Support for both Python and R code examples
    Specialized handling of statistical concepts and programming questions

Getting Started
Prerequisites

    Python 3.8+
    An OpenAI API key or Hugging Face account for embedding models

Installation

    Clone this repository
    Install dependencies:

    pip install -r requirements.txt

    Set up your API keys:

    export OPENAI_API_KEY=your-key-here

Usage

    Add your learning materials to the data/ directory
    Run the indexing script:

    python scripts/index_documents.py

    Start the chatbot:

    python scripts/run_chatbot.py

Project Structure

data-science-rag/
├── data/                      # Your learning materials
│   ├── statistics/
│   ├── python/
│   ├── r/
│   └── machine_learning/
├── vector_db/                 # The generated vector database
├── scripts/                   # Utility scripts
│   ├── index_documents.py     # Process and index documents
│   └── run_chatbot.py         # Interactive chatbot interface
├── src/                       # Core code
│   ├── __init__.py
│   ├── document_processor.py  # Document loading and chunking
│   ├── embedding.py           # Embedding models
│   ├── rag_engine.py          # RAG implementation
│   └── utils.py               # Utility functions
├── config.yml                 # Configuration file
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

Configuration

Edit config.yml to customize:

    Embedding model (OpenAI or Hugging Face)
    Document processing settings
    LLM parameters
    Retrieval settings

License

MIT
# daisy-rag-chatbot
