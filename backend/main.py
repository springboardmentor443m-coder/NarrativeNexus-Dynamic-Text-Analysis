from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from docx import Document
from text_preprocessing import preprocess_text
from semantic_topic_modeling import build_semantic_topics
from dataset_loader import analyze_twitter_financial_dataset
import uvicorn


app = FastAPI(title="Narrative Nexus Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    topics: dict
    tokens: list | None = None
    sample_size: int | None = None
    summary: str | None = None


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(text: str = Form(None), file: UploadFile | None = File(None)):
    content = ""
    if file:
        suffix = (file.filename or "").lower()
        if suffix.endswith(".txt"):
            content = (await file.read()).decode("utf-8", errors="ignore")
        elif suffix.endswith(".docx"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            with open(tmp.name, "wb") as f:
                f.write(await file.read())
            doc = Document(tmp.name)
            content = "\n".join(p.text for p in doc.paragraphs)

    if not content and text:
        content = text.strip()

    if not content:
        return AnalyzeResponse(topics={}, tokens=[])

    tokens = preprocess_text(content)
    result = build_semantic_topics([content])
    return AnalyzeResponse(topics=result["topics"], tokens=tokens, summary=result.get("summary"))


@app.get("/test-dataset")
async def test_dataset(limit: int = 1000):
    try:
        result = analyze_twitter_financial_dataset(limit)
        print("🔹 Backend Output Sample:", result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error loading dataset: {e}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

