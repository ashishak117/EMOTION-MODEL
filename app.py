import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import io

# 🔹 Disable GPU (Fix CUDA Errors on Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔹 Initialize Flask App
app = Flask(__name__)

# 🔹 Load Emotion Detection Model
MODEL_PATH = "emotion_detection_model_ver2.h5"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Emotion Labels (Ensure these match your trained model's output)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 🔹 Preprocessing Function
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# 🔹 Define API Route for Emotion Detection
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        processed_img = preprocess_image(file)
        prediction = model.predict(processed_img)
        emotion_index = np.argmax(prediction)  # Get the index of the highest prediction
        emotion_label = EMOTIONS[emotion_index]  # Map to emotion name

        return jsonify({"emotion": emotion_label})  # Return predicted emotion

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Run Flask App on Render's Required Port (5000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
