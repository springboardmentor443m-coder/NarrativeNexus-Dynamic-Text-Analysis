// app.js - lightweight frontend logic with client-side fallback summarizer

const inputText = document.getElementById('inputText');
const analyzeBtn = document.getElementById('analyzeBtn');
const summaryEl = document.getElementById('summary');
const sentimentEl = document.getElementById('sentiment');
const topicEl = document.getElementById('topic');
const fileInput = document.getElementById('fileInput');
const apiSelect = document.getElementById('apiSelect');

// ---------------- FILE UPLOAD ----------------
fileInput.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;

    // Only handle .txt and .csv on client
    if (f.name.endsWith('.txt') || f.name.endsWith('.csv')) {
        const txt = await f.text();
        inputText.value = txt;
    } else {
        alert(
            'Selected file: ' + f.name +
            '\n.docx client-side parsing is not implemented. Paste text manually or use backend.'
        );
    }
});

// ---------------- ANALYZE BUTTON ----------------
analyzeBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) {
        alert('Please paste or write some text first.');
        return;
    }

    // Loading placeholders
    summaryEl.textContent = 'Generating summary…';
    sentimentEl.textContent = '{\n "polarity": 0,\n "label": "neutral"\n}';
    topicEl.textContent = 'Predicting topic…';

    // ---------- 1) Try Local Backend ----------
    if (apiSelect.value.includes('Local API')) {
        try {
            const res = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!res.ok) throw new Error('Bad response');

            const data = await res.json();

            summaryEl.textContent =
                data.summary ?? fallbackSummary(text);

            sentimentEl.textContent =
                JSON.stringify(data.sentiment ?? simpleSentiment(text), null, 2);

            topicEl.textContent =
                data.topic ?? simpleTopic(text);

            return; // success
        } catch (err) {
            console.warn('Local API failed, falling back:', err);
        }
    }

    // ---------- 2) Fallback (client-side only) ----------
    summaryEl.textContent = fallbackSummary(text);
    sentimentEl.textContent = JSON.stringify(simpleSentiment(text), null, 2);
    topicEl.textContent = simpleTopic(text);
});

// ---------------- FALLBACK SUMMARY ----------------
function fallbackSummary(text, maxSentences = 2) {
    const sentences = text.match(/[^\.!\?]+[\.!\?]+|\s*[^\.!\?]+$/g) || [text];
    if (sentences.length <= maxSentences) return text;
    return sentences.slice(0, maxSentences).map(s => s.trim()).join(' ');
}

// ---------------- FALLBACK SENTIMENT ----------------
function simpleSentiment(text) {
    const pos = ['good','great','happy','excellent','positive','love','like','enjoy','nice'];
    const neg = ['bad','sad','terrible','hate','angry','worse','problem','difficult','poor'];

    const words = text.toLowerCase().match(/\b[a-z']+\b/g) || [];
    let score = 0;

    for (const w of words) {
        if (pos.includes(w)) score += 1;
        if (neg.includes(w)) score -= 1;
    }

    let label = 'neutral';
    if (score > 0) label = 'positive';
    if (score < 0) label = 'negative';

    return {
        polarity: Math.max(-1, Math.min(1, score / 10)),
        label
    };
}

// ---------------- FALLBACK TOPIC ----------------
function simpleTopic(text) {
    const stop = new Set([
        'the','and','a','an','of','in','on','to','is','it','that','this',
        'for','with','as','are','was','be','by','or','at','from'
    ]);

    const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
    const freq = {};

    for (const w of words) {
        if (stop.has(w)) continue;
        freq[w] = (freq[w] || 0) + 1;
    }

    const sorted = Object.keys(freq).sort((a, b) => freq[b] - freq[a]).slice(0, 6);

    if (sorted.length === 0) return 'text, analysis, topic';

    return sorted.join(', ');
}
