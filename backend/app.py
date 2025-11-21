from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib, json
from utils import preprocess_text, extractive_summary, predict_sentiment, match_topic
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dynamic Text Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load models (these are created by train_topics.py)
VEC = joblib.load("backend/models/vectorizer.joblib")
TM = joblib.load("backend/models/topic_model.joblib")
with open("backend/models/topics.json","r",encoding='utf-8') as f:
    TOPICS = json.load(f)

class PredictRequest(BaseModel):
    text: str

@app.post('/analyze')
async def analyze_text(req: PredictRequest):
    text = req.text or ""
    clean = preprocess_text(text)
    summary = extractive_summary(text)
    sentiment = predict_sentiment(text)
    topic = match_topic(clean, VEC, TM, TOPICS)
    return {"summary": summary, "sentiment": sentiment, "topic": topic}

@app.post('/upload_file')
async def upload_file(file: UploadFile = File(...)):
    contents = (await file.read()).decode(errors='ignore')
    return await analyze_text(PredictRequest(text=contents))

@app.get('/topics')
async def get_topics():
    return TOPICS
