from flask import Flask, request, jsonify

from src.inference import KeywordExtractor

# Initialize Flask app
app = Flask(__name__)

# Load model once at startup (global scope)
print("Loading NER model...")
extractor = KeywordExtractor(model_path="./output/my_keyword_model")
print("Model loaded successfully!")


@app.route("/extract", methods=["POST"])
def extract_keywords():
    """
    Extract keywords from text.

    Expects JSON: {"text": "user query here"}
    Returns JSON: {"keywords": ["keyword1", "keyword2", ...]}
    """
    # Get JSON data from request
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    if "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    text = data["text"]

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    # Extract keywords
    keywords = extractor.extract_keywords(text)

    return jsonify({"keywords": keywords})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "model_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

