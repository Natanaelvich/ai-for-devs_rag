from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_service import PDFRAGService
import os
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
rag_service = PDFRAGService()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.pdf'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        if not data["question"].strip():
            return jsonify({"error": "Question cannot be empty"}), 400
            
        question = data["question"]
        response, context = rag_service.ask(question)
        
        return jsonify({
            "response": response,
            "context": [doc.page_content for doc in context]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents", methods=["POST"])
def add_documents():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({"error": "No selected files"}), 400

        saved_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                saved_files.append(file_path)

        # Process the saved files
        result = rag_service.add_documents(saved_files, ALLOWED_EXTENSIONS)

        # Clean up uploaded files after processing
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except Exception:
                pass  # Ignore cleanup errors

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, port=5002) 