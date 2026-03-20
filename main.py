from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# CORS সমস্যা সমাধানের জন্য (যাতে আপনার সাইট থেকে এক্সেস পায়)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# আপনার মডেল লোড করা (Hugging Face থেকে সরাসরি লোড হবে)
model_path = "hellofoysal101/bangla-sentiment-bert"
classifier = pipeline("sentiment-analysis", model=model_path)

class TextData(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Bangla Sentiment API is Running!"}

@app.post("/predict")
async def predict(data: TextData):
    result = classifier(data.text)
    return result
