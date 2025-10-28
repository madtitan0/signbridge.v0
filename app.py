import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sign_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Build labels from your dataset folder (matches training order used by flow_from_directory)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
if os.path.isdir(DATASET_DIR):
    labels = sorted([d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d))])
else:
    # Fallback to A..Z
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Globals shared with /get_prediction
current_prediction = "..."
current_confidence = 0.0

def preprocess_frame(frame):
    # 64x64 RGB float32 [0,1]
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_frames():
    global current_prediction, current_confidence
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x = preprocess_frame(frame)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = labels[idx] if idx < len(labels) else str(idx)

        current_prediction = label
        current_confidence = conf

        cv2.putText(
            frame, f"Pred: {label} ({conf:.2f})", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        _, buffer = cv2.imencode(".jpg", frame)
        jpg = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route("/")
def index():
    return render_template("translate.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_prediction")
def get_prediction():
    return jsonify({"prediction": current_prediction,
                    "confidence": current_confidence})

if __name__ == "__main__":
    app.run(debug=True)
