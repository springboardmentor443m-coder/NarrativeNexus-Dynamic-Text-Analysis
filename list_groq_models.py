# DO NOT KEEP API KEY IN CODE!
# Use this instead:

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)
response = client.models.list()
print([m.id for m in response.data])

# backend/topic_modeler.py
"""
Adaptive TopicModeler:
- Balanced paragraph-based chunking (auto-adaptive).
- Stopword-light cleaning for chunks.
- BERTopic primary pipeline (if sentence-transformers / umap / hdbscan available).
- Dynamic embedding selection: general vs technical.
- Groq LLM JSON fallback and LLM-based topic renaming/refinement after BERTopic.
- Returns structured topics + diagnostics.
"""

from typing import Any, List, Dict, Optional, Union
import json, re

# Optional imports (graceful)
try:
    from bertopic import BERTopic  # type: ignore
    _BERTOPIC_OK = True
except Exception:
    _BERTOPIC_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_OK = True
except Exception:
    _SENTENCE_OK = False

try:
    from umap import UMAP  # type: ignore
    _UMAP_OK = True
except Exception:
    _UMAP_OK = False

try:
    import hdbscan  # type: ignore
    _HDBSCAN_OK = True
except Exception:
    _HDBSCAN_OK = False

# quick stopwords (small list to keep code light; extender-friendly)
_STOPWORDS = {
    "the","and","a","an","of","to","in","is","it","for","on","that","this","with","as","by","are","was","be","or","from"
}

def _safe_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

class TopicModeler:
    def __init__(
        self,
        groq_client: Optional[Any] = None,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_general: str = "all-mpnet-base-v2",
        embedding_technical: str = "all-distilroberta-v1",
        max_chunks_per_doc: int = 40,
        min_paragraph_len: int = 40,
    ):
        self.client = groq_client
        self.llm_model = llm_model
        self.emb_gen = embedding_general
        self.emb_tech = embedding_technical
        self.max_chunks_per_doc = int(max_chunks_per_doc)
        self.min_paragraph_len = int(min_paragraph_len)

        # prepare BERTopic if available
        self.topic_model = None
        self.embedder = None
        if _BERTOPIC_OK and _SENTENCE_OK:
            try:
                # default embedder will be created lazily based on content type
                self.topic_model = BERTopic(umap_model=(UMAP(n_neighbors=15, n_components=5, metric="cosine") if _UMAP_OK else None),
                                            hdbscan_model=(hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1) if _HDBSCAN_OK else None),
                                            top_n_words=8,
                                            nr_topics="auto",
                                            verbose=False)
            except Exception:
                self.topic_model = None

    # --- chunking & cleaning ---
    def _split_paragraphs(self, text: str) -> List[str]:
        # Primary split: double newlines
        paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
        if not paras:
            paras = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
        return paras

    def _merge_small(self, paras: List[str]) -> List[str]:
        res = []
        buffer = ""
        for p in paras:
            if len(p) < self.min_paragraph_len:
                buffer = (buffer + " " + p).strip() if buffer else p
                if len(buffer) >= self.min_paragraph_len:
                    res.append(buffer.strip())
                    buffer = ""
            else:
                if buffer:
                    res.append((buffer + " " + p).strip())
                    buffer = ""
                else:
                    res.append(p)
        if buffer:
            if res:
                res[-1] = (res[-1] + " " + buffer).strip()
            else:
                res.append(buffer)
        return res

    def _split_long_para(self, p: str) -> List[str]:
        if len(p) <= 2000:
            return [p]
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
        cur, out = [], []
        for s in sents:
            if len(" ".join(cur + [s])) > 800 and cur:
                out.append(" ".join(cur))
                cur = [s]
            else:
                cur.append(s)
        if cur:
            out.append(" ".join(cur))
        return out

    def _chunk_document(self, text: str) -> List[str]:
        paras = self._split_paragraphs(text)
        paras = self._merge_small(paras)
        chunks = []
        for p in paras:
            chunks.extend(self._split_long_para(p))
        # cap and merge evenly if too many
        if len(chunks) > self.max_chunks_per_doc:
            factor = (len(chunks) // self.max_chunks_per_doc) + 1
            merged = []
            for i in range(0, len(chunks), factor):
                merged.append(" ".join(chunks[i:i+factor]))
            chunks = merged
        # filter out extremely short chunks
        chunks = [c.strip() for c in chunks if len(c.strip()) >= 30]
        return chunks

    def _is_technical(self, text: str) -> bool:
        # heuristic: presence of code tokens, file extensions, common code words, many short tokens
        tech_markers = ["import ", "def ", "{", "}", "://", ".py", ".js", ".json", "SELECT ", "INSERT ", "HTTP/", "console.log", "==", "=>"]
        if any(tok in text for tok in tech_markers):
            return True
        # many short tokens or punctuation-heavy: likely technical/logs
        tokens = re.findall(r"\S+", text)
        short_ratio = sum(1 for t in tokens if len(t) <= 3) / (len(tokens) + 1)
        if short_ratio > 0.45 and len(tokens) > 40:
            return True
        return False

    def _clean_for_embedding(self, text: str) -> str:
        # light stopword removal (keeps meaning)
        words = re.findall(r"\w[\w\-']*\w|\w", text.lower())
        out = [w for w in words if w not in _STOPWORDS]
        return " ".join(out)

    # --- LLM fallback & refinement ---
    def _llm_topics_prompt(self, text: str) -> str:
        return (
            "Extract and return the main topics from the text as STRICT JSON only.\n"
            "Format: {\"topics\":[{\"topic_name\":\"label\",\"keywords\":[\"k1\",\"k2\"],\"count\":N}, ...]}\n"
            "Provide 3-12 topics depending on content. Topic names should be short (1-6 words), keywords 3-8 terms.\n\n"
            f"Text:\n{text[:9000]}"
        )

    def _llm_refine_topic_prompt(self, topic_name: str, keywords: List[str]) -> str:
        return (
            "Given a topic with keywords, produce a clean short human label and a refined keyword list (3-8 items).\n"
            "Return JSON: {\"topic_name\":\"label\",\"keywords\":[...]} and nothing else.\n\n"
            f"Current label: {topic_name}\nKeywords: {keywords}"
        )

    def _llm_fallback(self, text: str) -> Dict[str, Any]:
        if not self.client:
            return {"topics": [], "diagnostics": {"method": "llm_fallback", "status": "no_groq"}}
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": self._llm_topics_prompt(text)}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _safe_json(raw)
            if parsed and isinstance(parsed, dict) and parsed.get("topics"):
                out = []
                for i, t in enumerate(parsed["topics"], start=1):
                    nm = t.get("topic_name") or ", ".join(t.get("keywords", [])[:3])
                    kws = t.get("keywords", []) or []
                    cnt = int(t.get("count", 0)) if t.get("count") is not None else 0
                    out.append({"topic_id": f"LLM-{i}", "topic_name": nm, "keywords": kws, "count": cnt})
                return {"topics": out, "diagnostics": {"method": "llm_fallback", "status": "ok", "source_chunks": 1}}
        except Exception:
            pass
        return {"topics": [], "diagnostics": {"method": "llm_fallback", "status": "error"}}

    # --- public ---
    def extract_topics(self, text_or_docs: Union[str, List[str]]) -> Dict[str, Any]:
        # normalize
        if isinstance(text_or_docs, str):
            docs = [d.strip() for d in re.split(r"\n{2,}", text_or_docs) if d.strip()]
            if not docs:
                docs = [text_or_docs.strip()]
        else:
            docs = [d for d in text_or_docs if isinstance(d, str) and d.strip()]
        if not docs:
            return {"topics": [], "diagnostics": {"method": "none", "status": "no_text"}}

        # produce chunks per doc, keep track
        all_chunks = []
        for d in docs:
            chunks = self._chunk_document(d)
            if not chunks:
                chunks = [d.strip()]
            all_chunks.extend(chunks)

        if len(all_chunks) < 3 or not (self.topic_model and _SENTENCE_OK):
            # too small or missing deps -> LLM fallback
            combined = "\n\n".join(all_chunks)
            return self._llm_fallback(combined)

        # choose embedder dynamically
        combined_text = " ".join(all_chunks)
        tech = self._is_technical(combined_text)
        emb_model = self.emb_tech if tech else self.emb_gen

        try:
            self.embedder = SentenceTransformer(emb_model)
        except Exception:
            # embedding load failed -> fallback
            return self._llm_fallback(combined_text)

        try:
            # pre-clean chunks for embedding (light)
            cleaned_chunks = [self._clean_for_embedding(c) for c in all_chunks]
            embeddings = self.embedder.encode(cleaned_chunks, show_progress_bar=False)
            topics, probs = self.topic_model.fit_transform(cleaned_chunks, embeddings)
            info = self.topic_model.get_topic_info()
            final = []
            for _, row in info.iterrows():
                tid = int(row["Topic"])
                if tid == -1:
                    continue
                count = int(row.get("Count", 0))
                kwpairs = self.topic_model.get_topic(tid) or []
                keywords = [w for w, _ in kwpairs][:8]
                name = row.get("Name") if row.get("Name") and not str(row.get("Name")).startswith("Topic") else ", ".join(keywords[:3])
                # refine name + keywords with LLM (best-effort)
                refined = None
                if self.client:
                    try:
                        rp = self.client.chat.completions.create(
                            model=self.llm_model,
                            messages=[{"role": "system", "content": self._llm_refine_topic_prompt(name, keywords)}],
                            temperature=0.0,
                            max_tokens=120,
                        )
                        parsed = _safe_json(rp.choices[0].message.content.strip())
                        if parsed and parsed.get("topic_name"):
                            refined = {"topic_name": parsed.get("topic_name"), "keywords": parsed.get("keywords", keywords)}
                    except Exception:
                        refined = None
                if refined:
                    final.append({"topic_id": tid, "topic_name": refined["topic_name"], "keywords": refined["keywords"], "count": count})
                else:
                    final.append({"topic_id": tid, "topic_name": name, "keywords": keywords, "count": count})
            if not final:
                return self._llm_fallback(combined_text)
            return {"topics": final, "diagnostics": {"method": "bertopic", "status": "ok", "chunks": len(all_chunks), "technical_mode": tech}}
        except Exception:
            return self._llm_fallback(combined_text)
