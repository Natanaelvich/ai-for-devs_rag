from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_service import RAGService

load_dotenv()

app = Flask(__name__)
rag_service = RAGService()

VALID_METHODS = ["rerank", "parent"]

@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        if not data["question"].strip():
            return jsonify({"error": "Question cannot be empty"}), 400
        
        method = data.get("method", "rerank")
        if method not in VALID_METHODS:
            return jsonify({
                "error": f"Invalid method. Valid options are: {', '.join(VALID_METHODS)}"
            }), 400
            
        question = data["question"]
        response = rag_service.process_query(question, method)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, port=5000) 