function showLoading(targetId, message = "Processing...") {
    const el = document.getElementById(targetId);
    if (!el) return;
    el.innerHTML = `<span class="loading">${message}</span>`;
}

// ---------------- PREPROCESSING ----------------

async function runPreprocessing() {
    const text = document.getElementById("inputText").value.trim();
    if (!text) {
        alert("Please paste or type some text first.");
        return;
    }

    showLoading("preStats", "Cleaning text & counting tokens…");
    showLoading("cleanText");
    showLoading("tokens");

    try {
        const res = await fetch("/api/preprocess", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        document.getElementById("preStats").innerHTML = `
          <div class="stat-pill">
            <div class="stat-label">Original chars</div>
            <div class="stat-value">${data.original_chars}</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Cleaned chars</div>
            <div class="stat-value">${data.cleaned_chars}</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Words</div>
            <div class="stat-value">${data.words}</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Tokens</div>
            <div class="stat-value">${data.tokens}</div>
          </div>
        `;

        document.getElementById("cleanText").innerHTML = `
          <strong>Cleaned Text:</strong><br>
          <span>${data.cleaned_text}</span>
        `;

        const tokensHtml = (data.tokens_list || [])
            .map(t => `<span class="token-chip">${t}</span>`)
            .join("");

        document.getElementById("tokens").innerHTML = `
          <strong>Tokens:</strong>
          <div class="tokens-list">${tokensHtml}</div>
        `;
    } catch (err) {
        console.error(err);
        document.getElementById("preStats").innerHTML =
            `<span class="loading">Error while preprocessing.</span>`;
    }
}

// ---------------- SENTIMENT ----------------

function sentimentEmoji(label) {
    if (label === "Strong Positive" || label === "Positive") return "😊";
    if (label === "Strong Negative" || label === "Negative") return "☹️";
    return "😐";
}

async function runSentiment() {
    const text = document.getElementById("inputText").value.trim();
    if (!text) {
        alert("Please paste or type some text first.");
        return;
    }

    showLoading("sentimentOutput", "Running VADER sentiment model…");

    try {
        const res = await fetch("/api/sentiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        const label = data.label || "Neutral";
        const compound = data.compound ?? data.score ?? 0;
        const pos = data.pos ?? 0;
        const neu = data.neu ?? 0;
        const neg = data.neg ?? 0;
        const interpretation = data.interpretation ||
            "Overall emotional tone of the paragraph.";

        const position = ((compound + 1) / 2) * 100;
        const emoji = sentimentEmoji(label);

        document.getElementById("sentimentOutput").innerHTML = `
          <div class="sent-header-line">
              <div class="sent-label">
                Sentiment: ${label}
                <span class="sent-emoji">${emoji}</span>
              </div>
              <div class="sent-score">
                Compound score: <strong>${compound.toFixed(4)}</strong>
              </div>
          </div>

          <div class="sent-bar-wrapper">
              <div class="sent-bar-track">
                  <div class="sent-bar-marker" style="left: ${position}%;"></div>
              </div>
              <div class="sent-bar-labels">
                  <span>-1</span>
                  <span>Neutral</span>
                  <span>+1</span>
              </div>
          </div>

          <div class="sent-breakdown">
              <span>Positive: ${(pos * 100).toFixed(1)}%</span>
              <span>Neutral: ${(neu * 100).toFixed(1)}%</span>
              <span>Negative: ${(neg * 100).toFixed(1)}%</span>
          </div>

          <div class="sent-note">
              ${interpretation}
          </div>
        `;
    } catch (err) {
        console.error(err);
        document.getElementById("sentimentOutput").innerHTML =
            `<span class="loading">Error while running sentiment.</span>`;
    }
}

// ---------------- TOPIC MODELING ----------------

async function runTopicModel() {
    const text = document.getElementById("inputText").value.trim();
    if (!text) {
        alert("Please paste or type some text first.");
        return;
    }

    showLoading("topicResult", "Running trained NMF topic model…");

    try {
        const res = await fetch("/api/topics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        if (data.error) {
            document.getElementById("topicResult").innerHTML =
                `<span class="loading">${data.error}</span>`;
            return;
        }

        const topicName = data.topic_name || "Predicted Topic";
        const keywords = data.keywords || [];
        const topicId = data.topic_id;

        const kwHtml = keywords
            .map(k => `<span class="topic-keyword">${k}</span>`)
            .join("");

        document.getElementById("topicResult").innerHTML = `
          <div><strong>Topic ID:</strong> ${topicId}</div>
          <div style="margin-top:6px;">
            <strong>Predicted Topic Name:</strong><br>
            <span>${topicName}</span>
          </div>
          <div style="margin-top:6px;">
            <strong>Top Keywords:</strong>
            <div class="topic-keywords">${kwHtml}</div>
          </div>
          <div class="topic-note">
            Topic is inferred by applying a trained NMF model on the 20 Newsgroups dataset
            and matching your paragraph to the closest learned topic.
          </div>
        `;
    } catch (err) {
        console.error(err);
        document.getElementById("topicResult").innerHTML =
            `<span class="loading">Error while running topic model.</span>`;
    }
}

// ---------------- SUMMARY ----------------

async function runSummary() {
    const text = document.getElementById("inputText").value.trim();
    if (!text) {
        alert("Please paste or type some text first.");
        return;
    }

    showLoading("summaryOutput", "Building extractive summary…");

    try {
        const res = await fetch("/api/summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                type: document.getElementById("summaryType").value
            })
        });

        const data = await res.json();

        document.getElementById("summaryOutput").innerHTML = `
          <strong>Summary:</strong><br>
          <span>${data.summary}</span>
        `;
    } catch (err) {
        console.error(err);
        document.getElementById("summaryOutput").innerHTML =
            `<span class="loading">Error while generating summary.</span>`;
    }
}

// ---------------- FULL ANALYSIS & CLEAR ----------------

function runFullAnalysis() {
    runPreprocessing();
    runSentiment();
    runTopicModel();
    runSummary();
}

function clearAll() {
    document.getElementById("inputText").value = "";
    document.getElementById("preStats").innerHTML = "Waiting for input…";
    document.getElementById("cleanText").innerHTML = "Cleaned text will appear here.";
    document.getElementById("tokens").innerHTML = "Tokens will appear here.";
    document.getElementById("sentimentOutput").innerHTML = "Sentiment insights will appear here.";
    document.getElementById("topicResult").innerHTML = "Topic name and keywords will appear here.";
    document.getElementById("summaryOutput").innerHTML = "Summary will appear here.";
}
