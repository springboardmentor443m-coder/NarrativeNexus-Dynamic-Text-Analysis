from fastapi import FastAPI
from app.routes import  topic_routes
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# Access your GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Initialize FastAPI
app = FastAPI(
    title="NarrativeNexus API",
    version="1.0",
    description="Performs topic modeling, sentiment analysis, and summarization on text data."
)

# Include all route modules
app.include_router(topic_routes.router)

@app.get("/")
def root():
    """Root endpoint for quick API health check."""
    return {"message": "Welcome to NarrativeNexus API"}

# Allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*" for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)