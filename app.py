import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import cv2

# Disable GPU (to avoid CUDA errors on Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

app = Flask(__name__)

# Ensure the model file exists
MODEL_PATH = "face_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Emotion Labels (make sure these match your model’s outputs)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Preprocessing function
def preprocess_image(file_obj):
    # Read and convert image to grayscale
    image = Image.open(io.BytesIO(file_obj.read())).convert("L")
    # Resize image to 48x48 (or whatever your model expects)
    image = image.resize((48, 48))
    image = np.array(image) / 255.0  # Normalize
    # Add batch and channel dimensions: shape becomes (1, 48, 48, 1)
    image = np.expand_dims(image, axis=[0, -1])
    return image

@app.route("/")
def home():
    return "✅ Emotion Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        processed_img = preprocess_image(file)
        predictions = model.predict(processed_img)
        emotion_index = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_index]
        return jsonify({"emotion": emotion_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use the port provided by the environment (Render sets PORT)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
