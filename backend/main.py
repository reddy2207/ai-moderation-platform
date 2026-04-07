from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image

# -----------------------------
# INIT APP
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS (IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD TEXT MODEL
# -----------------------------
MODEL_PATH = "../ml/models/moderation_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cpu")
model.to(device)
model.eval()

# -----------------------------
# LOAD IMAGE MODEL
# -----------------------------
image_classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

# -----------------------------
# LABEL MAP
# -----------------------------
label_map = {
    0: "safe",
    1: "abuse",
    2: "toxic",
    3: "hate",
    4: "other"
}

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class TextInput(BaseModel):
    text: str

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def home():
    return {"message": "AI Moderation API Running 🚀"}

# -----------------------------
# TEXT PREDICTION
# -----------------------------
@app.post("/predict-text")
def predict_text(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    return {
        "label": label_map[predicted_class.item()],
        "confidence": float(confidence.item())
    }

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    results = image_classifier(image)

    return {
        "predictions": results
    }