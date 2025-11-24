import React, { useState } from "react";

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const backend = "http://127.0.0.1:5000";

  const analyzeText = async () => {
    if (!text) return;
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${backend}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setResult(await res.json());
    } catch (err) {
      setResult({ error: err.toString() });
    }
    setLoading(false);
  };

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${backend}/upload`, {
        method: "POST",
        body: form,
      });

      setResult(await res.json());
    } catch (err) {
      setResult({ error: err.toString() });
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 900, margin: "auto", padding: 20 }}>
      <h2>Narrative Nexus — Dynamic Text Analyzer</h2>

      <textarea
        placeholder="Paste text here…"
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", height: 140, padding: 10 }}
      />

      <button onClick={analyzeText} disabled={!text || loading}>
        Analyze Text
      </button>

      <br /> <br />

      <input
        type="file"
        accept=".txt,.csv,.docx,.pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={upload} disabled={!file || loading}>
        Upload & Analyze
      </button>

      {loading && <p>Processing…</p>}

      {result && !loading && (
        <div>
          {result.error && <pre>{result.error}</pre>}

          {/* Cleaned Text */}
          <h3>1. Preprocessing (Cleaned Text)</h3>
          {result.cleaned_text?.map((p, i) => (
            <p key={i} style={{ whiteSpace: "pre-wrap" }}>
              {p}
            </p>
          ))}

          {/* Sentiment */}
          <h3>2. Sentiment Analysis</h3>
          <pre>{JSON.stringify(result.sentiment, null, 2)}</pre>

          {/* Topics */}
          <h3>3. Topic Modeling</h3>
          <h4>LDA</h4>
          <pre>{JSON.stringify(result.lda_topics, null, 2)}</pre>
          <h4>NMF</h4>
          <pre>{JSON.stringify(result.nmf_topics, null, 2)}</pre>

          {/* Summaries */}
          <h3>4. Summarization</h3>
          {result.summaries?.map((s, i) => (
            <p key={i} style={{ whiteSpace: "pre-wrap" }}>
              {s}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
