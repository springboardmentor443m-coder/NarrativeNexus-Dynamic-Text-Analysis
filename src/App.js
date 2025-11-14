import React, {useState, useRef} from 'react';

export default function App(){
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const preprocessRef = useRef(null);
  const sentimentRef = useRef(null);
  const topicsRef = useRef(null);
  const summaryRef = useRef(null);

  const analyze = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      const data = await res.json();
      setResult(data);
      setTimeout(()=>{ if(preprocessRef.current) preprocessRef.current.scrollIntoView({behavior:'smooth'}); }, 200);
    } catch (e) {
      setResult({error: e.toString()});
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async () => {
    if(!file) return;
    setLoading(true);
    setResult(null);
    try{
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setResult(data);
      setTimeout(()=>{ if(preprocessRef.current) preprocessRef.current.scrollIntoView({behavior:'smooth'}); }, 200);
      setTimeout(()=>{ if(sentimentRef.current) sentimentRef.current.scrollIntoView({behavior:'smooth'}); }, 800);
      setTimeout(()=>{ if(topicsRef.current) topicsRef.current.scrollIntoView({behavior:'smooth'}); }, 1400);
      setTimeout(()=>{ if(summaryRef.current) summaryRef.current.scrollIntoView({behavior:'smooth'}); }, 2000);
    }catch(e){
      setResult({error: e.toString()});
    }finally{
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h2>Narrative Nexus — A Dynamic Text Analyzer</h2>
      <p>Paste text or upload a file (.txt, .csv, .docx, .pdf).</p>

      <div style={{marginBottom:12}}>
        <textarea value={text} onChange={e=>setText(e.target.value)} placeholder="Paste text here..."/>
        <div style={{marginTop:8}}>
          <button onClick={analyze} disabled={loading || !text}>Analyze Text</button>
        </div>
      </div>

      <div style={{marginTop:12}}>
        <input type="file" accept=".txt,.csv,.docx,.pdf" onChange={e=>setFile(e.target.files[0])} />
        <div style={{marginTop:8}}>
          <button onClick={uploadFile} disabled={loading || !file}>Upload & Analyze</button>
        </div>
      </div>

      {loading && <p>Analyzing…</p>}

      {result && (
        <div style={{marginTop:16}}>
          {result.error && <pre>{result.error}</pre>}

          {!result.error && (
            <div>
              <div className="section" ref={preprocessRef}>
                <h3>1. Preprocessing (Cleaned Text)</h3>
                {result.filename && <p><strong>File:</strong> {result.filename}</p>}
                {result.cleaned_text && (
  <p
    style={{
      whiteSpace: "pre-wrap",
      wordWrap: "break-word",
      overflowX: "hidden",
      lineHeight: "1.5",
      marginBottom: "0.75rem"
    }}
  >
    {result.cleaned_text}
  </p>
)}
              </div>

              <div className="section" ref={sentimentRef}>
                <h3>2. Sentiment Analysis</h3>
                {result.sentiments && result.sentiments.map((s,i)=>(<pre key={i}>{JSON.stringify(s,null,2)}</pre>))}
              </div>

              <div className="section" ref={topicsRef}>
                <h3>3. Topic Modeling</h3>
                <h4>LDA Topics</h4>
                <pre>{JSON.stringify(result.lda_topics,null,2)}</pre>
                <h4>NMF Topics</h4>
                <pre>{JSON.stringify(result.nmf_topics,null,2)}</pre>
              </div>

              <div className="section" ref={summaryRef}>
                <h3>4. Summarization</h3>
                {result.summaries.map((s, i) => (<p key={i} style={{ whiteSpace: "pre-wrap", wordWrap: "break-word", lineHeight: "1.5" }}>{s}</p>))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
