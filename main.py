from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline
from pydantic import BaseModel
import os

app = FastAPI()

# CORS সমস্যা সমাধানের জন্য
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# মডেল লোড করার পাথ (আপনার নতুন আপলোড করা কোয়ান্টাইজড ফাইল)
model_id = "hellofoysal101/bangla-sentiment-bert"

# মেমোরি বাঁচাতে আমরা ONNX Runtime ব্যবহার করব
try:
    print("Loading quantized model...")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, 
        file_name="model_quantized.onnx"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # পাইপলাইন তৈরি
    classifier = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class TextData(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Quantized Bangla Sentiment API is Running!"}

@app.post("/predict")
async def predict(data: TextData):
    # ইনপুট টেক্সট নিয়ে প্রেডিকশন
    result = classifier(data.text)
    return result
