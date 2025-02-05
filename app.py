from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("emotion_detection_model_ver2.h5")  # Ensure the model file is in the same directory

# Define class labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))
    
    # Convert image to grayscale and resize (modify as per your model input)
    image = np.array(image.convert("L").resize((48, 48)))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dimensions

    # Make prediction
    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    predicted_emotion = EMOTIONS[emotion_index]

    return jsonify({"emotion": predicted_emotion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
