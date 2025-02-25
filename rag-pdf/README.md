# PDF RAG Service

This service provides a REST API interface for querying PDF documents using RAG (Retrieval Augmented Generation).

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Service

Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5002`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the health status of the service.

### Add PDF Documents
```
POST /api/documents
Content-Type: multipart/form-data
```

Upload PDF files to be indexed and queried. Only PDF files are supported.

### Query Documents
```
POST /api/query
Content-Type: application/json

{
    "question": "Your question about the PDF content here"
}
```

Example questions:
- "Quais os principais riscos do marco legal de ia?"
- "Como o marco legal define inteligência artificial?"
- "Quais são as penalidades previstas para violações do marco legal de IA?"

Example response:
```json
{
    "response": "Generated answer based on the PDF content",
    "context": [
        "Text chunk from the PDF that was used to generate the answer...",
        "Another text chunk from the PDF..."
    ]
}
```

## Error Handling

The API will return appropriate error messages with corresponding HTTP status codes:
- 400: Bad Request (missing or invalid parameters)
- 500: Internal Server Error (processing errors)

## Implementation Details

The service uses:
- LangChain for document processing and retrieval
- OpenAI embeddings for vector search
- ChromaDB as the vector store
- PyPDF for PDF parsing

## Default Document

The service comes with a default document (`DOC-SF238339076816-20230503.pdf`) that will be automatically indexed if present in the root directory. 