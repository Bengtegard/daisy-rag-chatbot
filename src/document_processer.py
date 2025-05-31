"""
Document processor for loading and chunking documents.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers.models.gpt2 import GPT2TokenizerFast

from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    NotebookLoader
)


def remove_invalid_unicode(text: str) -> str:
    """Removes invalid surrogate pairs like \ud835"""
    return text.encode('utf-16', 'replace').decode('utf-16', 'replace')

class DocumentProcessor:
    """Handles document loading and chunking for various file types."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.supported_extensions = config.get('supported_extensions', ['.pdf', '.txt', '.csv', '.ipynb', '.R', '.py', '.md'])
        
        # Initialize GPT2 tokenizer for token counting
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def token_length(text: str) -> int:
            return len(self.tokenizer.encode(text))
        
        self.token_length = token_length
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.token_length,
            separators=[
                "\n\n",        # Paragraph breaks
                "\n# ",        # Section headers
                "\n## ",       # Subsection headers
                ". ",          # End of sentences with period
                "! ",          # End of sentences with exclamation
                "? ",          # End of sentences with question
                "\n",          # Line breaks
                " ",           # Word breaks
                ""             # Character breaks
            ]
        )

        self.extension_to_loader = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.py': TextLoader,
            '.ipynb': NotebookLoader,
            '.R': TextLoader,
        }

    def load_documents(self, data_dir: str) -> List[Document]:
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"Warning: Data directory {data_dir} does not exist")
            return documents

        for file_path in data_path.glob('**/*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.extension_to_loader:
                    try:
                        loader_cls = self.extension_to_loader[ext]
                        loader = loader_cls(str(file_path))
                        file_docs = loader.load()

                        for doc in file_docs:
                            rel_path = file_path.relative_to(data_path)
                            doc.metadata['rel_path'] = str(rel_path)
                            doc.metadata['category'] = file_path.parent.name
                            doc.metadata['source'] = str(file_path)

                        documents.extend(file_docs)
                        print(f"Loaded {len(file_docs)} documents from {file_path}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        return documents

    def load_web_pages(self, urls: List[str]) -> List[Document]:
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                web_docs = loader.load()
                for doc in web_docs:
                    doc.metadata['category'] = 'web'
                    doc.metadata['url'] = url
                documents.extend(web_docs)
                print(f"Loaded content from {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return documents

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks based primarily on folder structure."""
        all_chunks = []

        for doc in documents:
            doc.page_content = remove_invalid_unicode(doc.page_content)
            category = doc.metadata.get('category', '').lower()
            file_ext = Path(doc.metadata.get('source', '')).suffix.lower()
            
            # Create chunkers with appropriate separators
            if category in ['python']:
                chunks = self._chunk_programming_content(doc, language="python")
            elif category in ['r']:
                chunks = self._chunk_programming_content(doc, language="r")
            elif category in ['sql']:
                chunks = self._chunk_programming_content(doc, language="sql")
            elif category in ['statistics', 'linear_algebra', 'data_science', 'machine learning']:
                chunks = self._chunk_technical_content(doc)
            # Chunking based on file extensions
            elif file_ext == '.ipynb':
                chunks = self._chunk_notebook_content(doc)
            elif file_ext == '.html':
                chunks = self._chunk_html_content(doc)
            elif file_ext in ['.md', '.rmd']:
                chunks = self._chunk_markdown_content(doc)
            else:
                chunks = self.text_splitter.split_documents([doc])

            # Simple post-processing to fix obvious sentence splits
            chunks = self._simple_fix_sentences(chunks)

            for chunk in chunks:
                chunk.metadata['content_category'] = category

            all_chunks.extend(chunks)

        for i, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_id'] = i
            tokens = self.token_length(chunk.page_content)
            if tokens > self.chunk_size:
                print(f"Warning: Chunk {i} token count {tokens} exceeds chunk_size {self.chunk_size}")

        return all_chunks
        
    def _simple_fix_sentences(self, chunks: List[Document]) -> List[Document]:
        """Simple fix for sentences split across chunks."""
        if len(chunks) <= 1:
            return chunks
            
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # If current chunk doesn't end with punctuation and next starts with lowercase
            current_content = current_chunk.page_content.rstrip()
            next_content = next_chunk.page_content.lstrip()
            
            if (current_content and next_content and
                not current_content.endswith(('.', '!', '?', ':', ';')) and
                next_content[0].islower()):
                
                # Take the first sentence from the next chunk
                first_sentence_end = -1
                for end_marker in ['. ', '! ', '? ']:
                    pos = next_content.find(end_marker)
                    if pos >= 0 and (first_sentence_end == -1 or pos < first_sentence_end):
                        first_sentence_end = pos + len(end_marker)
                
                if first_sentence_end > 0:
                    # Move the first sentence of next chunk to the current chunk
                    current_chunk.page_content = current_content + " " + next_content[:first_sentence_end]
                    next_chunk.page_content = next_content[first_sentence_end:]
                else:
                    split_point = min(len(next_content), 50)
                    current_chunk.page_content = current_content + " " + next_content[:split_point]
                    next_chunk.page_content = next_content[split_point:]
        
        return chunks
    
    def _chunk_programming_content(self, doc: Document, language: str) -> List[Document]:
        if language == "python":
            code_patterns = [
                "def ", "class ", "import ", "from ", "print(", "return ",
                "if ", "elif ", "else:", "for ", "while ", "try:", "except ",
                "with ", "async ", "await ", "lambda ", "yield ",
                "plt.", "np.", "pd.", "tf.", "keras",
                "self.", "@staticmethod", "@classmethod", "@property"
            ]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=self.token_length,
                separators=[
                    "\n## ", "\n# ", "\nChapter ", "\n\n",
                    "\nIn[", "\nOut[", "```python", "```\n",
                    ". ", "! ", "? ",
                    "\n", " ", ""
                ]
            )
            
            chunks = splitter.split_documents([doc])
            
            for chunk in chunks:
                if any(pattern in chunk.page_content for pattern in code_patterns):
                    chunk.metadata['contains_code'] = True
                    chunk.metadata['language'] = language
                    
            return chunks

        elif language == "r":
            code_patterns = ["function(", "<-", "library(", "install.packages("]
            special_separators = ["\n## ", "\n# ", "```r\n", "\n```"]
            separators = special_separators + ["\n\n\n", "\n\n", ". ", "! ", "? ", "\n", " ", ""]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=self.token_length,
                separators=separators
            )
            
            chunks = splitter.split_documents([doc])
            
            for chunk in chunks:
                if any(pattern in chunk.page_content for pattern in code_patterns):
                    chunk.metadata['contains_code'] = True
                    chunk.metadata['language'] = language
            
            return chunks
        
        elif language == "sql":
            code_patterns = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ", "DROP "]
            special_separators = [
                "\n-- ",        # SQL comment line
                "\n/*",         # SQL block comment start
                "*/\n",         # SQL block comment end
                "\n;\n",        # Statement delimiter
                "\n\n\n", "\n\n", ". ", "! ", "? ", "\n", " ", ""
            ]
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=self.token_length,
                separators=special_separators
            )
            
            chunks = splitter.split_documents([doc])

            for chunk in chunks:
                if any(pattern in chunk.page_content for pattern in code_patterns):
                    chunk.metadata['contains_code'] = True
                    chunk.metadata['language'] = language
            
            return chunks

        else:
            # Fallback for unknown languages
            code_patterns = []
            special_separators = []
            separators = special_separators + ["\n\n\n", "\n\n", ". ", "! ", "? ", "\n", " ", ""]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250,
                length_function=self.token_length,
                separators=separators
            )
            
            chunks = splitter.split_documents([doc])
            
            for chunk in chunks:
                if any(pattern in chunk.page_content for pattern in code_patterns):
                    chunk.metadata['contains_code'] = True
                    chunk.metadata['language'] = language
            
            return chunks

    def _chunk_technical_content(self, doc: Document) -> List[Document]:
        code_patterns = ["def ", "import ", "class ", "=", "print(", "plt.", "np.", "pd.", "tf.", "keras"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self.token_length,
            separators=[
                # Book/paper structure
                "\nChapter ", "\nSection ", "\n## ", "\n# ",
                # Equations and math blocks
                "\n$", "$\n", "\n$", "$\n",
                # Code blocks (Python in technical books)
                "```python", "```\n", "\n```",
                "\ndef ", "\nclass ", "\nimport ", "\nfrom ",
                # Examples and exercises
                "\nExample ", "\nExercise ", "\nProblem ",
                # Regular text breaks
                "\n\n\n", "\n\n", ". ", "! ", "? ", "\n", " ", ""
            ]
        )
        chunks = splitter.split_documents([doc])

        for chunk in chunks:
            if any(pattern in chunk.page_content for pattern in code_patterns):
                chunk.metadata['contains_code'] = True
                chunk.metadata['language'] = "python"

        return chunks


    def _chunk_notebook_content(self, doc: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self.token_length,
            separators=["```\n", "\n```", "\n## ", "\n# ", "\n\n", ". ", "! ", "? ", "\n", " "]
        )
        return splitter.split_documents([doc])

    def _chunk_markdown_content(self, doc: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self.token_length,
            separators=["## ", "# ", "\n\n", ". ", "! ", "? ", "\n", " "]
        )
        chunks = splitter.split_documents([doc])

        if doc.metadata.get('source', '').lower().endswith('.rmd'):
            for chunk in chunks:
                if any(r_marker in chunk.page_content for r_marker in ['```{r', 'library(', '<-']):
                    chunk.metadata['contains_code'] = True
                    chunk.metadata['language'] = 'r'

        return chunks

    def _chunk_html_content(self, doc: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            length_function=self.token_length,
            separators=["</h1>", "</h2>", "</h3>", "</div>", "<br>", "<p>", "</p>", "\n\n", ". ", "! ", "? ", "\n", " "]
        )
        return splitter.split_documents([doc])