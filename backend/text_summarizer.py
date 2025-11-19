# backend/text_summarizer.py
import json, re
from typing import List, Dict
from groq import Groq


class TextSummarizer:
    """
    Simple per-file summarizer.
    - No combined summary
    - Ensures clean, complete sentences
    - Strong JSON output enforcement
    """

    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile", chunk_size: int = 1500):
        self.client = groq_client
        self.model = model
        self.chunk_size = int(chunk_size)

    # ---------------------- HELPERS ----------------------
    def _safe_json(self, raw: str):
        raw = raw.strip()
        try:
            return json.loads(raw)
        except:
            m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
            if m:
                try: return json.loads(m.group(1))
                except: return None
        return None

    def _chunks(self, text: str) -> List[str]:
        """Split long text into LLM-safe chunks."""
        words = text.split()
        chunks, buf = [], []
        for w in words:
            if len(" ".join(buf + [w])) > self.chunk_size:
                chunks.append(" ".join(buf))
                buf = [w]
            else:
                buf.append(w)
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    def _chunk_prompt(self, chunk: str) -> str:
        return (
            "Summarize the following text into 1–3 clear, complete sentences. "
            "Return STRICT JSON only: {\"summary\": \"...\"}.\n\n"
            f"Text:\n{chunk}"
        )

    def _file_prompt(self, file_name: str, all_chunk_summaries: List[str]) -> str:
        joined = "\n".join(all_chunk_summaries)
        return (
            "Combine the following chunk summaries into one clean, polished paragraph. "
            "Use full sentences only. STRICT JSON ONLY:\n"
            "{\"file_name\": \"<name>\", \"summary\": \"...\"}\n\n"
            f"File name: {file_name}\n\nChunk summaries:\n{joined}"
        )

    def _run_llm(self, prompt: str, max_tokens: int = 250) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return r.choices[0].message.content.strip()

    # ---------------------- PUBLIC API ----------------------
    def summarize_documents(self, docs: List[str], file_names: List[str]) -> List[Dict[str, str]]:
        """
        Returns ONLY per-file summaries.
        Output:
        [
          {"file_name": "doc1.pdf", "summary": "..."},
          {"file_name": "notes.txt", "summary": "..."}
        ]
        """
        results = []

        for doc, fname in zip(docs, file_names):
            chunks = self._chunks(doc)
            chunk_summaries = []

            # summarize each chunk
            for ch in chunks:
                raw = self._run_llm(self._chunk_prompt(ch), max_tokens=180)
                parsed = self._safe_json(raw)
                if parsed and parsed.get("summary"):
                    chunk_summaries.append(parsed["summary"].strip())
                else:
                    # fallback: ensure clean ending
                    s = raw.strip()
                    if s and s[-1] not in ".!?": s += "."
                    chunk_summaries.append(s)

            # combine chunk summaries into one per-file summary
            raw_final = self._run_llm(self._file_prompt(fname, chunk_summaries), max_tokens=250)
            parsed_final = self._safe_json(raw_final)

            if parsed_final and parsed_final.get("summary"):
                summary = parsed_final["summary"].strip()
            else:
                summary = " ".join(chunk_summaries)
                if summary and summary[-1] not in ".!?": summary += "."

            results.append({"file_name": fname, "summary": summary})

        return results
