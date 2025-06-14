# Daisy - An Data Science RAG Assistant

A personalized Retrieval-Augmented Generation (RAG) chatbot for data science learning materials. 

## Demo
![image](https://github.com/user-attachments/assets/c7f808b2-2ed6-4a78-8b30-cf00c769f695)

### Response example when asking about the regularization technique "dropout"
![image](https://github.com/user-attachments/assets/74b961c0-eb1f-426a-bffe-49c95da5d28d)


## Overview

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
    For this repo Groq API is used for the deepseek-r1-distill-llama-70b model

Installation

    Clone this repository
    Install dependencies:

    pip install -r requirements.txt

    Set up your API keys:

    export OPENAI_API_KEY=your-key-here

Usage

    Add your learning materials to the data/ directory

    Run the document_processor.py
    
    Create the FAISS vector database

    Start the streamlit app:

    streamlit run scripts/rag_ui.py

    The notebook shows a detailed step by step pipeline to get started.

Project Structure

    data-science-rag/
    ├── data/                      # learning materials
    │   ├── data_science
    │   ├── linear_algebra
    │   ├── machine_learning
    │   ├── personal_notes
    │   ├── python/
    │   ├── r/
    │   └── sql/
    │   ├── statistics/
    ├── vector_db/                 # The generated vector database
    ├── scripts/                   # Utility scripts
    │   └── rag_ui.py              # Interactive chatbot interface with streamlit
    ├── notebooks/                 # Pipeline for trying src code and evalute model
    ├── src/                       
    │   ├── __init__.py
    │   ├── document_processor.py  # Document loading and chunking
    │   ├── math_render.py         # Renders math equation to LateX
    │   ├── rag_engine.py          # RAG implementation
    |   ├── token_manager.py       # Token limit manager
    │   └── utils.py               # Utility functions
    ├── config.yml                 # Configuration file
    ├── requirements.txt           # Dependencies
    └── README.md                  # Project documentation

Configuration

Edit config.yml to customize:

    Document processing settings
    LLM parameters
    Retrieval settings
