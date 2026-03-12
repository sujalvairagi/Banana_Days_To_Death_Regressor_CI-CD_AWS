import os
import uuid
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy-load prediction pipeline (loads models on first request)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from bananaPredictor.pipeline.prediction import BananaPredictionPipeline
        _predictor = BananaPredictionPipeline()
    return _predictor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a file with key 'image'."}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Save to unique temp path to avoid collisions
    filename = secure_filename(image.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    temp_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        image.save(temp_path)

        predictor = get_predictor()
        result = predictor.predict(temp_path)

        # Encode annotated image as base64 for response
        annotated = result.pop("annotated_image", None)
        annotated_b64 = None
        if annotated is not None:
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        response = {
            "total_bananas": result["total_bananas"],
            "predictions": result["predictions"],
            "summary": result["summary"],
            "annotated_image_base64": annotated_b64,
        }

        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
