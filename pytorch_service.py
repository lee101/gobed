#!/usr/bin/env python3
"""
PyTorch native model service - provides embeddings via HTTP API.
This bypasses ONNX conversion entirely.
"""

import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global model instance
model = None

def load_model():
    """Load the SentenceTransformer model."""
    global model
    if model is None:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
        print("Model loaded successfully!")

@app.route('/embed', methods=['POST'])
def embed_text():
    """Embed a single text or list of texts."""
    try:
        data = request.get_json()
        
        if 'text' in data:
            # Single text
            text = data['text']
            embedding = model.encode(text)
            return jsonify({
                'embedding': embedding.tolist(),
                'shape': list(embedding.shape)
            })
        elif 'texts' in data:
            # Multiple texts
            texts = data['texts']
            embeddings = model.encode(texts)
            return jsonify({
                'embeddings': embeddings.tolist(),
                'shape': list(embeddings.shape)
            })
        else:
            return jsonify({'error': 'Missing text or texts field'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/compare', methods=['POST'])
def compare_embeddings():
    """Compare embeddings between native PyTorch and tokenized input."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        token_ids = data.get('token_ids', [])
        
        # Get native embedding
        native_embedding = model.encode(text)
        
        # Get embedding from token IDs if provided
        if token_ids:
            # This is more complex - we'd need to access the model's internals
            # For now, just return the native embedding
            return jsonify({
                'native_embedding': native_embedding.tolist(),
                'text': text,
                'message': 'Token ID embedding not implemented yet'
            })
        
        return jsonify({
            'native_embedding': native_embedding.tolist(),
            'text': text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("Starting PyTorch native model service on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
