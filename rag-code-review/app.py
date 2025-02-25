from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_service import CodeReviewRAGService
import os
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
rag_service = CodeReviewRAGService()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.ipynb'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/review", methods=["POST"])
def review_code():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        if not data["question"].strip():
            return jsonify({"error": "Question cannot be empty"}), 400
            
        question = data["question"]
        response = rag_service.process_query(question)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/repository", methods=["POST"])
def add_repository():
    try:
        data = request.get_json()
        if not data or "repository_url" not in data:
            return jsonify({"error": "Missing 'repository_url' in request body"}), 400
        
        repository_url = data["repository_url"]
        branch = data.get("branch", "main")
        
        result = rag_service.add_repository(repository_url, branch)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/code", methods=["POST"])
def add_code_files():
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
        result = rag_service.add_code_files(saved_files, ALLOWED_EXTENSIONS)

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
    app.run(debug=True, port=5001) 