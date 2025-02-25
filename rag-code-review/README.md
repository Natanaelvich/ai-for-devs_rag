# Code Review RAG Service

This service provides a REST API interface for code review using RAG (Retrieval Augmented Generation). It can analyze code repositories and provide suggestions for improvements.

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

The server will start on `http://localhost:5001`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns the health status of the service.

### Add Repository
```
POST /api/repository
Content-Type: application/json

{
    "repository_url": "https://github.com/username/repository",
    "branch": "main"  // optional, defaults to "main"
}
```

Clones a Git repository and indexes its code for review. This will replace any previously indexed repository.

### Add Code Files
```
POST /api/code
Content-Type: multipart/form-data
```

Upload individual code files for review. Supported file extensions: `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.cs`, `.go`, `.rb`, `.php`, `.ipynb`.

### Request Code Review
```
POST /api/review
Content-Type: application/json

{
    "question": "Your code review question here"
}
```

Example questions:
- "Você pode revisar e sugerir melhorias para o código?"
- "Identifique possíveis problemas de segurança no código"
- "Como posso melhorar a performance deste código?"

Example response:
```json
{
    "response": "Generated code review based on the code context"
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
- GitPython for repository management 