"""
Flask Application untuk NER Deployment
Aplikasi web untuk Named Entity Recognition
"""

from flask import Flask, request, jsonify, render_template
from model_utils import get_model
import os
import sys
import traceback
import json

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./models/ner_models"
app.config['JSON_AS_ASCII'] = False  # Support Unicode (Indonesian text)

# Label mapping untuk normalisasi display
# Model kamu: B-ORGANISATION, B-PERSON, B-PLACE, I-ORGANISATION, I-PERSON, I-PLACE, O
LABEL_MAPPING = {
    'PERSON': 'PER',
    'ORGANISATION': 'ORG',  # British spelling
    'ORGANIZATION': 'ORG',  # US spelling (fallback)
    'PLACE': 'LOC',
    'LOCATION': 'LOC',
}

# Color mapping untuk frontend
COLOR_MAPPING = {
    'PER': '#ff9999',   # Pink untuk Person
    'ORG': '#99ccff',   # Blue untuk Organization
    'LOC': '#99ff99',   # Green untuk Location/Place
}

# ============================================================
# Load Model on Startup
# ============================================================
print("="*70)
print("🚀 STARTING NER FLASK APPLICATION")
print("="*70)
print(f"📂 Model path: {MODEL_PATH}")

# Check if model exists
model = None
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model folder not found at {MODEL_PATH}")
    print(f"📝 Please make sure you have:")
    print(f"   1. Trained the model or downloaded from Colab")
    print(f"   2. Model files are in: {MODEL_PATH}/")
    print(f"   3. Required files: config.json, pytorch_model.bin, tokenizer files")
else:
    try:
        model = get_model(MODEL_PATH)
        print("="*70)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print(f"📊 Total labels: {len(model.id2label)}")
        print(f"🏷  Labels: {list(model.label2id.keys())}")
        print("="*70)
    except Exception as e:
        print("="*70)
        print(f"❌ ERROR LOADING MODEL: {e}")
        print("="*70)
        traceback.print_exc()
        model = None


# ============================================================
# Helper Functions
# ============================================================

def normalize_label(label):
    """
    Normalize label untuk konsistensi display
    Contoh: PERSON -> PER, ORGANISATION -> ORG, PLACE -> LOC
    """
    return LABEL_MAPPING.get(label, label)


def format_entities_for_frontend(text, entities):
    """
    Convert entities ke format yang dibutuhkan frontend
    Format: list of {text: str, label: str}
    
    Args:
        text: Original text
        entities: List of entities dari model.predict()
                  Format: [{"text": "...", "label": "...", "start": int, "end": int}]
    
    Returns:
        List of {text: str, label: str} untuk frontend
    """
    result = []
    last_end = 0
    
    for entity in entities:
        # Add text before entity (label 'O')
        if entity['start'] > last_end:
            before_text = text[last_end:entity['start']]
            if before_text:
                result.append({
                    'text': before_text,
                    'label': 'O'
                })
        
        # Add entity with normalized label
        normalized_label = normalize_label(entity['label'])
        result.append({
            'text': entity['text'],
            'label': normalized_label
        })
        
        last_end = entity['end']
    
    # Add remaining text after last entity
    if last_end < len(text):
        remaining_text = text[last_end:]
        if remaining_text:
            result.append({
                'text': remaining_text,
                'label': 'O'
            })
    
    return result


def generate_highlighted_html(text, entities):
    """
    Generate HTML dengan entities yang di-highlight
    
    Args:
        text: Original text
        entities: List of entities
        
    Returns:
        HTML string dengan highlighted entities
    """
    if not entities:
        return f'<span>{text}</span>'
    
    html = ''
    last_end = 0
    
    for entity in entities:
        # Text before entity
        if entity['start'] > last_end:
            html += f'<span>{text[last_end:entity["start"]]}</span>'
        
        # Entity with highlight
        normalized_label = normalize_label(entity['label'])
        color = COLOR_MAPPING.get(normalized_label, '#cccccc')
        html += f'<span class="entity" style="background-color: {color}; padding: 2px 5px; border-radius: 3px; margin: 0 2px;" title="{normalized_label}">{entity["text"]}</span>'
        
        last_end = entity['end']
    
    # Text after last entity
    if last_end < len(text):
        html += f'<span>{text[last_end:]}</span>'
    
    return html


# ============================================================
# WEB INTERFACE ROUTES
# ============================================================

# app.py (Contoh route home yang diperbaiki)
@app.route("/", methods=["GET", "POST"]) # <--- IZINKAN POST
def home():
    raw_text = ""
    highlighted_html = ""
    
    if model is None:
        # Jika model gagal load saat startup
        return render_template("index.html", error="Model gagal dimuat. Cek log server."), 503

    if request.method == "POST":
        text = request.form.get("text_input")
        if text:
            # PENTING: Panggil fungsi prediksi yang benar (misalnya ner_prediction_function)
            # Karena Anda tidak menyertakan logic highlight di app.py ini, 
            # asumsikan model.predict(text) mengembalikan list entitas, 
            # lalu Anda harus memanggil generate_highlighted_html(text, entities)
            
            entities = model.predict(text) # Asumsi model memiliki method predict()
            highlighted_html = generate_highlighted_html(text, entities)
            raw_text = text # Kirim kembali teks asli

        return render_template("index.html", raw_text=raw_text, prediction_html=highlighted_html)
    
    # Untuk GET request
    return render_template("index.html")


# ============================================================
# API ROUTES
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint
    """
    if model:
        return jsonify({
            "status": "online",
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "total_labels": len(model.id2label),
            "labels": list(model.label2id.keys()),
            "device": str(model.device)
        })
    else:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "message": "Model not loaded. Check server logs."
        }), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analyze endpoint untuk frontend
    Format response: list of {text: str, label: str}
    """
    print(f"\n{'='*70}")
    print(f"📨 POST /analyze - Request received")
    
    if not model:
        print("❌ Model not loaded")
        return jsonify({
            "error": "Model not loaded. Please check server configuration."
        }), 500
    
    try:
        data = request.get_json()
        print(f"📥 Data received: {data}")
        
        if not data or "text" not in data:
            print("❌ Missing 'text' field")
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data["text"]
        print(f"📝 Text to analyze ({len(text)} chars): {text[:100]}...")
        
        if not text or not text.strip():
            print("❌ Empty text")
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        if len(text) > 5000:
            print("❌ Text too long")
            return jsonify({
                "error": "Text too long. Maximum 5000 characters."
            }), 400
        
        # Get entities from model
        print("🔍 Running prediction...")
        entities = model.predict(text)
        print(f"✅ Found {len(entities)} entities")
        
        # Debug: print entities
        for entity in entities:
            print(f"   - {entity['text']:20s} | {entity['label']:15s} | [{entity['start']}:{entity['end']}]")
        
        # Convert to frontend format (list of tokens with labels)
        formatted_result = format_entities_for_frontend(text, entities)
        
        print(f"✅ Returning {len(formatted_result)} tokens")
        print(f"{'='*70}\n")
        
        return jsonify(formatted_result)
    
    except Exception as e:
        print(f"❌ Error in /analyze: {e}")
        traceback.print_exc()
        print(f"{'='*70}\n")
        return jsonify({
            "error": f"Analysis failed: {str(e)}"
        }), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict endpoint (detailed format)
    Format response: {success: bool, text: str, entities: list, highlighted_html: str}
    """
    print(f"\n{'='*70}")
    print(f"📨 POST /predict - Request received")
    
    if not model:
        print("❌ Model not loaded")
        return jsonify({
            "error": "Model not loaded. Please check server configuration."
        }), 500
    
    try:
        data = request.get_json()
        print(f"📥 Data: {data}")
        
        if not data or "text" not in data:
            print("❌ Missing 'text' field")
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data["text"]
        print(f"📝 Text to predict: {text[:100]}...")
        
        if not text or not text.strip():
            print("❌ Empty text")
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        if len(text) > 5000:
            print("❌ Text too long")
            return jsonify({
                "error": "Text too long. Maximum 5000 characters."
            }), 400
        
        # Predict entities
        print("🔍 Running prediction...")
        entities = model.predict(text)
        
        # Normalize labels in entities for consistent display
        normalized_entities = []
        for entity in entities:
            normalized_entity = entity.copy()
            normalized_entity['label_original'] = entity['label']
            normalized_entity['label'] = normalize_label(entity['label'])
            normalized_entities.append(normalized_entity)
        
        # Generate highlighted HTML
        highlighted_html = generate_highlighted_html(text, entities)
        
        print(f"✅ Found {len(entities)} entities")
        for entity in normalized_entities:
            print(f"   - {entity['text']:20s} | {entity['label']:5s} ({entity['label_original']:15s})")
        print(f"{'='*70}\n")
        
        # Return result
        return jsonify({
            'success': True,
            'text': text,
            'entities': normalized_entities,
            'count': len(normalized_entities),
            'highlighted_html': highlighted_html
        })
    
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        traceback.print_exc()
        print(f"{'='*70}\n")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict NER for multiple texts
    """
    print(f"\n{'='*70}")
    print(f"📨 POST /api/batch_predict - Request received")
    
    if not model:
        print("❌ Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or "texts" not in data:
            print("❌ Missing 'texts' field")
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        texts = data["texts"]
        
        if not isinstance(texts, list):
            print("❌ 'texts' must be a list")
            return jsonify({"error": "'texts' must be a list"}), 400
        
        if len(texts) > 100:
            print("❌ Batch too large")
            return jsonify({"error": "Batch size too large. Maximum 100 texts."}), 400
        
        print(f"📝 Processing {len(texts)} texts...")
        
        # Use model's predict_batch method
        results = model.predict_batch(texts)
        
        # Normalize labels in results
        for result in results:
            for entity in result['entities']:
                entity['label_original'] = entity['label']
                entity['label'] = normalize_label(entity['label'])
        
        print(f"✅ Processed {len(results)} texts")
        print(f"{'='*70}\n")
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(results)
        })
    
    except Exception as e:
        print(f"❌ Error in /api/batch_predict: {e}")
        traceback.print_exc()
        print(f"{'='*70}\n")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labels", methods=["GET"])
def get_labels():
    """Get all available labels"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get both original and normalized labels
    original_labels = list(model.label2id.keys())
    normalized_labels = list(set([normalize_label(label) if label != 'O' else 'O' 
                                   for label in original_labels]))
    
    return jsonify({
        "success": True,
        "labels": normalized_labels,
        "original_labels": original_labels,
        "count": len(original_labels),
        "label_mapping": LABEL_MAPPING
    })


@app.route("/model-info", methods=["GET"])
@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Get informasi tentang model"""
    if not model:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Load metrics jika ada
    metrics_path = os.path.join(MODEL_PATH, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            metrics = {}
    
    # Load evaluation results jika ada
    eval_path = os.path.join(MODEL_PATH, "evaluation_results.json")
    eval_results = {}
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
        except:
            eval_results = {}
    
    return jsonify({
        'success': True,
        'model_path': MODEL_PATH,
        'device': str(model.device),
        'labels': list(model.label2id.keys()),
        'label_count': len(model.label2id),
        'label_mapping': LABEL_MAPPING,
        'metrics': metrics,
        'evaluation': eval_results
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    print(f"❌ 404 Error: {request.path}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    print(f"❌ 405 Error: {request.method} {request.path}")
    return jsonify({
        "error": "Method not allowed",
        "allowed_methods": ["GET", "POST"]
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"❌ 500 Error: {error}")
    traceback.print_exc()
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    if model is None:
        print("\n⚠  WARNING: Starting server without loaded model!")
        print("🔧 Server will run but prediction endpoints will not work.")
        print("📝 Please check model files and restart the server.\n")
    
    print("\n" + "="*70)
    print("🌐 FLASK SERVER STARTING...")
    print("="*70)
    print("📍 Local URL:   http://127.0.0.1:5000")
    print("📍 Network URL: http://0.0.0.0:5000")
    print("\n💡 Available Endpoints:")
    print("   GET  /                    - Web interface")
    print("   POST /analyze             - Analyze text (frontend format)")
    print("   POST /predict             - Predict entities (detailed)")
    print("   GET  /api/health          - Health check")
    print("   POST /api/predict         - Single text prediction")
    print("   POST /api/batch_predict   - Batch prediction")
    print("   GET  /api/labels          - Get all labels")
    print("   GET  /api/model-info      - Get model information")
    print("\n🛑 Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=True
    )