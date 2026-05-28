from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from groq import Groq
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
app = Flask(__name__, static_folder='dist', static_url_path='/')
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔥 Set Tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------
# Load model ONCE
# -----------------------------
model = models.mobilenet_v2(weights="DEFAULT")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet labels
import urllib.request
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    categories = [line.strip().decode('utf-8') for line in f.readlines()]

# -----------------------------
# Image Classification
# -----------------------------
def classify_image(file_storage):
    image = Image.open(file_storage).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted = torch.max(output, 1)
    return categories[predicted[0]]

# -----------------------------
# PDF Extraction
# -----------------------------
def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# -----------------------------
# OCR
# -----------------------------
def extract_text_from_image(file_stream):
    image = Image.open(file_stream)
    return pytesseract.image_to_string(image).strip()

# -----------------------------
# Universal File Handler
# -----------------------------
def extract_text_from_any(file_storage):
    filename = file_storage.filename.lower()
    file_storage.stream.seek(0)

    # Image → classify
    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        try:
            pred = classify_image(file_storage)
            return f"Image shows: {pred} (Not a medical diagnosis)"
        except:
            file_storage.stream.seek(0)
            return extract_text_from_image(file_storage.stream)

    # PDF
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file_storage.stream)

    # DOCX
    if filename.endswith('.docx'):
        try:
            from docx import Document
            file_storage.stream.seek(0)
            doc = Document(file_storage.stream)
            return "\n".join([p.text for p in doc.paragraphs])
        except:
            return "Could not read DOCX file"

    # TXT
    if filename.endswith('.txt'):
        try:
            file_storage.stream.seek(0)
            return file_storage.read().decode('utf-8', errors='ignore')
        except:
            return "Could not read TXT file"

    return "Unsupported file type"

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        pred = classify_image(file)
        return jsonify({
            "diagnosis": f"Image shows: {pred} (Not medical advice)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    history = request.form.get('history', '[]')

    try:
        history = json.loads(history)
    except:
        history = []

    file = request.files.get('file')

    if file:
        extracted = extract_text_from_any(file)
        user_message += "\n" + extracted

    if not user_message:
        return jsonify({"response": "Send message or file"})

    history.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            temperature=0.7,
            max_tokens=800
        )
        reply = completion.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        return jsonify({
            "response": reply,
            "history": history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)