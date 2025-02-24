# RAG API Service

This service provides a REST API interface for querying documents using RAG (Retrieval Augmented Generation) with two different approaches: reranking and parent document retrieval.

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
COHERE_API_KEY=your_cohere_api_key
```

## Running the Service

Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the health status of the service.

### Query Document
```
POST /api/query
Content-Type: application/json

{
    "question": "Your question here",
    "method": "rerank"  // or "parent"
}
```

The `method` parameter is optional and defaults to "rerank". Available options are:
- `rerank`: Uses semantic search with Cohere reranking
- `parent`: Uses parent document retrieval approach

Example response:
```json
{
    "response": "Generated answer based on the document context"
}
```

## Error Handling

The API will return appropriate error messages with corresponding HTTP status codes:
- 400: Bad Request (missing or invalid parameters)
- 500: Internal Server Error (processing errors)

## Document Storage

The service uses two vector stores:
- `naiveDB/`: Stores document chunks for the reranking approach
- `chieldVectorDB/`: Stores document chunks for the parent document approach

These directories are automatically created when the service starts. 