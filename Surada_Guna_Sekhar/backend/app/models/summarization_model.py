"""
Summarization module for AI-Narrative-Nexus
Uses Google Gemini LLM for abstractive summarization.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("⚠️ GEMINI_API_KEY not found in environment variables")

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    GEMINI_READY = True
    print("✅ Gemini model initialized for summarization")
except Exception as e:
    print(f"⚠️ Failed to initialize Gemini model: {e}")
    GEMINI_READY = False


class TextSummarizer:
    """Calls Gemini to generate a single meaningful summary."""

    def summarize(self, text: str) -> dict:
        """
        Generate a concise, meaningful summary using Gemini.

        Returns:
            dict with:
            - summary
            - status ('success' or 'error')
            - error (if any)
        """
        if not GEMINI_READY:
            return {
                "summary": "",
                "status": "error",
                "error": "Gemini model not initialized",
            }

        if not text or len(text.strip()) == 0:
            return {
                "summary": "",
                "status": "error",
                "error": "Empty text",
            }

        try:
            if len(text.split()) < 30:
                return {
                    "summary": text,
                    "status": "success",
                }

            prompt = (
                "You are an AI assistant for a dynamic text analysis platform.\n"
                "The following text has already been preprocessed (cleaned and normalized).\n"
                "Generate a clear, concise paragraph that captures the main ideas, key themes, "
                "and overall message of the text. Do not copy long sentences verbatim; "
                "use your own wording and keep it 3–6 sentences.\n\n"
                "Preprocessed text:\n"
                f"{text}\n\n"
                "Summary:"
            )

            response = model.generate_content(prompt)
            summary_text = response.text.strip()

            return {
                "summary": summary_text,
                "status": "success",
            }

        except Exception as e:
            return {
                "summary": "",
                "status": "error",
                "error": str(e),
            }


text_summarizer = TextSummarizer()
