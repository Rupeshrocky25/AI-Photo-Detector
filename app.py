from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
import numpy as np
import cv2
import pillow_heif
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
LOCAL_MODEL_PATH = "./model"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "heic"}

AI_THRESHOLD = 0.60
REAL_THRESHOLD = 0.60
BLUR_THRESHOLD = 140.0

# --------------------------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH).to(device)
model.eval()

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(img, max_dim=1024):
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def blur_score(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def looks_like_ai_art(img):
    arr = np.array(img)
    color_std = np.std(arr, axis=(0,1)).mean()
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()
    blur = blur_score(img)

    return color_std < 45 and edge_density > 0.07 and blur < 180

def make_preview(img):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ---------------- CORE LOGIC ----------------
def predict_image(img):
    img = resize_image(img)

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    real_prob = ai_prob = 0.0
    for idx, label in model.config.id2label.items():
        lname = label.lower()
        if "real" in lname:
            real_prob = float(probs[idx])
        elif "ai" in lname or "fake" in lname:
            ai_prob = float(probs[idx])

    blur = blur_score(img)
    oversmoothed = blur < BLUR_THRESHOLD

    if looks_like_ai_art(img):
        label = "AI"
    elif ai_prob >= AI_THRESHOLD:
        label = "AI"
    elif real_prob >= REAL_THRESHOLD:
        label = "Real"
    else:
        label = "AI" if oversmoothed else "Real"

    return {
        "label": label,
        "probabilities": {
            "AI": round(ai_prob, 3),
            "Real": round(real_prob, 3)
        },
        "blur_score": round(blur, 2),
        "oversmoothed": oversmoothed
    }

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"})

        if file.filename.lower().endswith(".heic"):
            heif = pillow_heif.read_heif(file.read())
            img = Image.frombytes(heif.mode, heif.size, heif.data).convert("RGB")
        else:
            img = Image.open(file).convert("RGB")

        result = predict_image(img)
        result["preview"] = make_preview(img)
        return jsonify(result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
