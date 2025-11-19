from typing import Any, List, Dict, Optional, Union
import json, re

# imports 
_BERT=True; _SENT=True; _UMAP=True; _HDB=True
try: from bertopic import BERTopic
except: _BERT=False
try: from sentence_transformers import SentenceTransformer
except: _SENT=False
try: from umap import UMAP
except: _UMAP=False
try: import hdbscan
except: _HDB=False

# small stopword list → improves keyword quality
_STOP = {"the","and","a","an","of","to","in","is","it","for","on",
         "that","this","with","as","by","are","was","be","or","from"}

# safe LLM JSON parser (handles extra text)
def _parse_json(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    try: return json.loads(raw)
    except:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        return json.loads(m.group(0)) if m else None


class TopicModeler:
    def __init__(
        self,
        groq_client: Optional[Any] = None,
        llm_model: str = "llama-3.3-70b-versatile",
        embedding_general: str = "all-mpnet-base-v2",
        embedding_technical: str = "all-distilroberta-v1",
        embedding_model_name: Optional[str] = None,  # backward compatibility
        max_chunks_per_doc: int = 40,
        min_paragraph_len: int = 40
    ):
        self.client = groq_client
        self.llm_model = llm_model
        self.emb_gen = embedding_model_name or embedding_general
        self.emb_tech = embedding_technical
        self.max_chunks = max_chunks_per_doc
        self.min_para = min_paragraph_len

        # build BERTopic if available
        self.topic_model = None
        if _BERT:
            try:
                umap = UMAP(n_neighbors=15, n_components=5, metric="cosine") if _UMAP else None
                hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1) if _HDB else None
                self.topic_model = BERTopic(
                    umap_model=umap,
                    hdbscan_model=hdb,
                    top_n_words=8,
                    nr_topics="auto",
                    verbose=False
                )
            except:
                self.topic_model = None

        self.embedder = None

    # ----------------------------- chunking -----------------------------
    def _paras(self, text: str) -> List[str]:
        """Split by paragraphs; fallback to sentence split."""
        p = [x.strip() for x in re.split(r"\n{2,}", text) if x.strip()]
        return p or [x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]

    def _chunks_from(self, text: str) -> List[str]:
        """Merge too-small paragraphs; split overly long ones."""
        paras = self._paras(text)
        merged, buf = [], ""
        for p in paras:
            if len(p) < self.min_para:
                buf = (buf + " " + p).strip()
                if len(buf) >= self.min_para: merged.append(buf); buf=""
            else:
                merged.append((buf+" "+p).strip() if buf else p); buf=""
        if buf: merged.append(buf)

        chunks=[]
        for m in merged:
            if len(m)<=2000: chunks.append(m); continue
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', m) if s.strip()]
            cur=[]
            for s in sents:
                if cur and len(" ".join(cur+[s]))>800: chunks.append(" ".join(cur)); cur=[s]
                else: cur.append(s)
            if cur: chunks.append(" ".join(cur))

        # compress if too many chunks
        if len(chunks)>self.max_chunks:
            step=(len(chunks)//self.max_chunks)+1
            chunks=[" ".join(chunks[i:i+step]) for i in range(0,len(chunks),step)]
        return [c for c in chunks if len(c)>=30]

    # --------------------------- utilities -----------------------------
    def _is_technical(self, text: str) -> bool:
        """Detect if text is code-heavy → choose better embedding model."""
        markers=["import ","def ","{","}","//",".py",".js",".json","SELECT ","HTTP/"]
        if any(m in text for m in markers): return True
        toks=re.findall(r"\S+",text)
        return (sum(1 for t in toks if len(t)<=3)/(len(toks)+1))>0.45 and len(toks)>40

    def _clean(self, s: str) -> str:
        """Light stopword removal for embeddings."""
        ws = re.findall(r"[A-Za-z0-9\-']+", s.lower())
        return " ".join([w for w in ws if w not in _STOP])

    def _target(self, n: int) -> int:
        """Dynamic topic count based on corpus size."""
        return 4 if n<5 else (10 if n<25 else 15)

    # --------------------------- LLM prompts ---------------------------
    def _refine_prompt(self, label, kws):
        return ( "RETURN JSON {\"topic_name\":\"...\",\"keywords\":[...]}. "
                 "Give short human topic name.\n"
                 f"Label: {label}\nKeywords: {', '.join(kws[:12])}" )

    def _extract_prompt(self, text, target):
        return ( f"Extract exactly {target} topics. "
                 "STRICT JSON: {\"topics\":[{\"topic_name\":\"...\",\"keywords\":[...],\"count\":N},...]}\n\n"
                 f"Text:\n{text[:8000]}" )

    # ----------------------------- LLM calls -----------------------------
    def _llm_refine(self, label, kws):
        if not self.client: return None
        try:
            r=self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"system","content":self._refine_prompt(label,kws)}],
                temperature=0.0, max_tokens=120
            )
            return _parse_json(r.choices[0].message.content)
        except: return None

    def _llm_extract(self, text, target):
        """Used when BERTopic can't run."""
        if not self.client:
            return {"topics":[],"diagnostics":{"method":"llm_fallback","status":"no_client"}}
        try:
            short=text[:2000] + ("\n\n"+text[-2000:] if len(text)>4000 else "")
            r=self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"system","content":self._extract_prompt(short,target)}],
                temperature=0.0,max_tokens=350
            )
            parsed=_parse_json(r.choices[0].message.content)
            if parsed and parsed.get("topics"):
                out=[]
                for i,t in enumerate(parsed["topics"],1):
                    lab=t.get("topic_name") or ", ".join(t.get("keywords",[])[:3])
                    out.append({
                        "topic_id":f"LLM-{i}",
                        "topic_name":lab,
                        "keywords":t.get("keywords",[]),
                        "count":int(t.get("count",0))
                    })
                return {"topics":out,"diagnostics":{"method":"llm_fallback","status":"ok"}}
        except: pass
        return {"topics":[],"diagnostics":{"method":"llm_fallback","status":"error"}}

    # --------------------------- main entry point ---------------------------
    def extract_topics(self, text_or_docs):
        """Public API — returns topic JSON + diagnostics."""
        docs = ([d.strip() for d in re.split(r"\n{2,}",text_or_docs) if d.strip()]
                if isinstance(text_or_docs,str)
                else [d for d in text_or_docs if isinstance(d,str) and d.strip()])
        if not docs: return {"topics":[],"diagnostics":{"method":"none","status":"no_text"}}

        chunks=[]
        for d in docs: chunks.extend(self._chunks_from(d) or [d])
        n=len(chunks); target=self._target(n); combined=" ".join(chunks)

        # too small or no BERTopic → LLM fallback
        if n<3 or not(self.topic_model and _SENT):
            return self._llm_extract(combined,target)

        # embed chunks
        try:
            emb=self.emb_tech if self._is_technical(combined) else self.emb_gen
            self.embedder=SentenceTransformer(emb)
            cleaned=[self._clean(c) for c in chunks]
            embeds=self.embedder.encode(cleaned,show_progress_bar=False)
        except:
            return self._llm_extract(combined,target)

        # BERTopic
        try:
            topics,probs=self.topic_model.fit_transform(cleaned,embeds)
            info=self.topic_model.get_topic_info()
            extracted=[]
            for _,row in info.iterrows():
                tid=int(row["Topic"])
                if tid==-1: continue
                cnt=int(row.get("Count",0))
                kwp=self.topic_model.get_topic(tid) or []
                kws=[w for w,_ in kwp][:8]
                label=row.get("Name") if row.get("Name") and not str(row["Name"]).startswith("Topic") else ", ".join(kws[:3])
                ref=self._llm_refine(label,kws)
                extracted.append({
                    "topic_id":tid,
                    "topic_name":ref["topic_name"] if ref else label,
                    "keywords":ref.get("keywords",kws) if ref else kws,
                    "count":cnt
                })
            extracted=sorted(extracted,key=lambda x: x["count"],reverse=True)[:target]
            return {"topics":extracted,"diagnostics":{"method":"bertopic","status":"ok","chunks":n}}
        except:
            return self._llm_extract(combined,target)
