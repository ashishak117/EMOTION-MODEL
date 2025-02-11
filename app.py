import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
import io
from PIL import Image

# 🔹 Disable GPU to avoid CUDA errors on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")  # Ensures TensorFlow does not use GPU

# 🔹 Initialize Flask App
app = Flask(__name__)

# 🔹 Load Emotion Detection Model
MODEL_PATH = "emotion_detection_model_ver2.h5"  # Ensure this file is in the root directory

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Emotion Labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 🔹 Preprocessing Function
def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.convert("L").resize((48, 48))  # Convert to grayscale & resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dimensions
    return image

# 🔹 API Route for Emotion Detection
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        processed_img = preprocess_image(file)
        predictions = model.predict(processed_img)
        emotion_index = np.argmax(predictions)  # Get the index of the highest prediction
        emotion_label = EMOTIONS[emotion_index]  # Get emotion name

        return jsonify({"emotion": emotion_label})  # Return predicted emotion

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Run Flask App on Render's Required Port (5000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
