"""
TEXT SUMMARIZER
Creates concise summaries of text.
Uses Groq LLM for intelligent summarization.
"""

from groq import Groq
from typing import Dict


class TextSummarizer:
    """
    Generates summaries of text documents.
    Simple and straightforward.
    """
    
    def __init__(self, groq_client: Groq, model_name: str):
        """
        Initialize the summarizer.
        
        groq_client: Groq API client
        model_name: Groq model to use
        """
        self.client = groq_client
        self.model = model_name
    
    def summarize(self, text: str, max_sentences: int = 5) -> str:
        """
        Create a summary of the text.
        
        text: Text to summarize
        max_sentences: How many sentences in the summary
        
        Returns: Summary text
        """
        instructions = (
            f"Summarize this text in {max_sentences} clear sentences. "
            f"Focus on the main points. Just the summary, no extra text."
        )
        
        try:
            # Limit to 3000 characters to avoid API limits
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": f"{instructions}\n\n{text[:3000]}"}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Could not generate summary: {str(e)}"