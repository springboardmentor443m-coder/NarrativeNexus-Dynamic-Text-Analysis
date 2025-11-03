from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from docx import Document
from text_preprocessing import preprocess_text
from topic_modeling import build_lda_model
from dataset_loader import analyze_twitter_financial_dataset
from text_analysis import clean_summarize_analyze
from dataset_text_analysis import analyze_twitter_financial_dataset_text
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    topics: list
    tokens: list
    coherence: float | None = None

@app.post('/analyze', response_model=AnalyzeResponse)
async def analyze(text: str = Form(None), file: UploadFile | None = File(None)):
    content = ''
    if file:
        suffix = (file.filename or '').lower()
        if suffix.endswith('.txt'):
            content = (await file.read()).decode('utf-8', errors='ignore')
        elif suffix.endswith('.docx'):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
            with open(tmp.name, 'wb') as f:
                f.write(await file.read())
            doc = Document(tmp.name)
            content = '\n'.join(p.text for p in doc.paragraphs)

    if not content and text:
        content = text

    if not content:
        return AnalyzeResponse(topics=[], tokens=[], coherence=None)

    tokens = preprocess_text(content)
    result = build_lda_model([tokens], num_topics=3)

    return AnalyzeResponse(
        topics=result['topics'],
        tokens=tokens,
        coherence=result['coherence']
    )

@app.get("/test-dataset")
async def test_dataset(limit: int = 2000):
    """Loads and analyzes the Twitter financial news dataset from Hugging Face."""
    result = analyze_twitter_financial_dataset(limit=limit)
    return result

@app.post("/clean-and-summarize")
async def clean_and_summarize(file: UploadFile = File(...)):
    """Accepts a text or HTML file, cleans, summarizes, and performs sentiment analysis."""
    content = (await file.read()).decode("utf-8", errors="ignore")
    result = clean_summarize_analyze(content)
    return result

@app.get("/analyze-hf-dataset")
async def analyze_hf_dataset(limit: int = 100):
    """
    Run cleaning, summarization, and sentiment analysis on Hugging Face dataset samples.
    """
    results = analyze_twitter_financial_dataset_text(limit=limit)
    return {"sample_size": len(results), "results": results}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
