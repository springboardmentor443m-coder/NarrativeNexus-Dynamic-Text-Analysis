import json
import re
import time
from typing import List, Dict, Any
from groq import Groq

class TextSummarizer:
    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = model
        
        # --- RATE LIMIT CONFIGURATION ---
        # Groq Free Tier often caps at ~6,000 to 15,000 TPM (Tokens Per Minute).
        # 1 token approx 4 chars.
        # We set a safe limit for "Full Mode" to avoid hitting TPM instantly.
        self.full_mode_limit = 15000  # ~3,750 tokens (Safe for single request)
        
        # Chunk size for larger files
        self.chunk_size = 8000        # ~2,000 tokens per chunk
        
        # Delay between chunks to respect RPM/TPM limits
        self.rate_limit_delay = 2     # Seconds

    def _clean_json_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Extracts JSON from LLM response, handling Markdown code blocks.
        """
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            
            match = re.search(r"(\{.*\})", raw_response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            return {}

    def _create_chunks(self, text: str) -> List[str]:
        """Splits text into chunks based on character limit."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1 
            
            if current_length >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def summarize_text(self, text: str) -> str:
        """
        Smart summarization with Rate Limit protection.
        """
        if not text:
            return "No text provided."

        # --- STRATEGY 1: FULL CONTEXT (For smaller docs) ---
        if len(text) < self.full_mode_limit:
            print(f"Processing in Full Mode ({len(text)} chars)...")
            prompt = (
                "Summarize the following text into a cohesive executive summary. "
                "Return ONLY valid JSON: {\"summary\": \"YOUR SUMMARY HERE\"}\n\n"
                f"TEXT:\n{text}"
            )
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert summarizer. Output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=600
                )
                data = self._clean_json_response(completion.choices[0].message.content)
                return data.get("summary", "Summary generation failed.")
            except Exception as e:
                return f"Error: {str(e)}"

        # --- STRATEGY 2: CHUNKING (For larger docs) ---
        print(f"Processing in Chunk Mode ({len(text)} chars)...")
        chunks = self._create_chunks(text)
        intermediate_summaries = []

        for i, chunk in enumerate(chunks):
            # Rate Limit Protection
            if i > 0: 
                time.sleep(self.rate_limit_delay)

            prompt = (
                "Summarize this section in 2 sentences. Capture main points. "
                "Return ONLY valid JSON: {\"summary\": \"...\"}\n\n"
                f"TEXT: {chunk}"
            )
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=250
                )
                data = self._clean_json_response(completion.choices[0].message.content)
                if data.get("summary"):
                    intermediate_summaries.append(data["summary"])
            except Exception as e:
                print(f"Chunk error: {e}")
                continue

        if not intermediate_summaries:
            return "Could not generate summary."

        # Final Synthesis
        combined_text = " ".join(intermediate_summaries)
        
        # Final synthesis might still be large, so we handle that carefully
        final_prompt = (
            "Create a final executive summary from these notes. "
            "Return ONLY valid JSON: {\"summary\": \"...\"}\n\n"
            f"NOTES: {combined_text[:25000]}" # Hard cap to prevent overflow
        )

        try:
            time.sleep(1) # Safety pause before final call
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert editor. Output JSON only."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            data = self._clean_json_response(completion.choices[0].message.content)
            return data.get("summary", combined_text)
            
        except Exception as e:
            return combined_text