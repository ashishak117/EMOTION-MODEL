import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model_path = "emotion_detection_model_ver2.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model file {model_path} not found!")

# Emotion Labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route("/")
def home():
    return "✅ Emotion Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    
    image = np.array(image.convert("L").resize((48, 48)))  # Convert to grayscale & resize
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dimensions

    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    predicted_emotion = EMOTIONS[emotion_index]

    return jsonify({"emotion": predicted_emotion})

# Run Flask app on Railway-assigned port
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
