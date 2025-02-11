import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# 🔹 Force CPU for compatibility & performance
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🔹 Initialize Flask App
app = Flask(__name__)

# 🔹 Load Model
MODEL_PATH = "emotion_detection_model_ver2.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file NOT FOUND! Make sure 'emotion_detection_model_ver2.h5' is uploaded.")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 🔹 Emotion Labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 🔹 Preprocessing Function
def preprocess_image(img):
    img = Image.open(io.BytesIO(img.read())).convert("L")  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Add batch & channel dimensions
    return img

# 🔹 API Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        processed_img = preprocess_image(file)
        prediction = model.predict(processed_img)
        emotion_index = np.argmax(prediction)
        emotion_label = EMOTIONS[emotion_index]

        return jsonify({"emotion": emotion_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Ensure Render Binds to the Correct Port
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=True)
