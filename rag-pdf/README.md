# PDF RAG System

This is a Retrieval Augmented Generation (RAG) system for PDF documents. It allows you to process PDF files, index their content, and query them using natural language.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

The system consists of two main scripts:

1. `pdf_rag.py`: Contains the core PDF processing functionality
2. `query_pdf.py`: Implements the RAG querying system

### Processing and Querying a PDF

To process and query a PDF, run:

```bash
python query_pdf.py
```

This will:
1. Process the PDF file and split it into chunks
2. Create embeddings for each chunk
3. Store the embeddings in a vector store
4. Allow you to query the content using natural language

### Example Usage in Python

```python
from query_pdf import PDFQueryEngine

# Initialize the engine
engine = PDFQueryEngine()

# Process a PDF
engine.process_and_index_pdf("your_document.pdf")

# Query the processed content
question = "What is the main topic of this document?"
answer = engine.query(question)
print(answer)
```

## Features

- PDF text extraction
- Text chunking with configurable size and overlap
- Embedding generation with caching
- Vector store indexing using FAISS
- Natural language querying
- Context-aware responses

## Requirements

The main dependencies are:
- langchain
- PyPDF2
- faiss-cpu
- openai

See `requirements.txt` for the complete list of dependencies. 