// -------------------------------------------------
// ANALYZE BUTTON CLICK EVENT
// -------------------------------------------------

document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value.trim();
    const modelType = document.getElementById("modelType").value;
    const summaryMode = document.getElementById("summaryMode").value;
    const nTopics = document.getElementById("nTopics").value;
    const nWords = document.getElementById("nWords").value;

    if (!text) {
        showError("Please enter some text before analyzing.");
        return;
    }

    loading(true);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                model_type: modelType,
                summary_mode: summaryMode,
                n_topics: nTopics,
                n_words: nWords
            })
        });

        const data = await response.json();
        console.log("DEBUG API RESPONSE:", data);

        loading(false);

        if (data.error) {
            showError(data.error);
            return;
        }

        renderResults(data);

    } catch (error) {
        loading(false);
        showError("Server error. Please try again.");
        console.error(error);
    }
});


// -------------------------------------------------
// UTILITY: Convert object → array safely
// -------------------------------------------------
function toArray(obj) {
    if (Array.isArray(obj)) return obj;               // already array
    if (typeof obj === "string") return obj.split(","); // string → array
    if (typeof obj === "object") return Object.values(obj); // object → array
    return [];                                        // fallback
}


// -------------------------------------------------
// RENDER API RESULTS INTO UI
// -------------------------------------------------

function renderResults(data) {

    // ------------------- Preprocessing Stats --------------------
    document.getElementById("statOriginalChars").textContent = data.stats.original_chars;
    document.getElementById("statCleanedChars").textContent = data.stats.cleaned_chars;
    document.getElementById("statOriginalWords").textContent = data.stats.word_count;
    document.getElementById("statTokens").textContent = data.stats.token_count;
    document.getElementById("statUniqueTokens").textContent = data.stats.unique_tokens;

    // ------------------- Sentiment --------------------
    document.getElementById("sentimentLabel").textContent = data.sentiment.label || "Neutral";
    document.getElementById("sentimentScore").textContent =
        `Score: ${Number(data.sentiment.score).toFixed(3)}`;

    // ------------------- Predicted Topic --------------------
    if (data.topic_prediction) {
        const topicID = data.topic_prediction.topic_id;
        let keywords = toArray(data.topic_prediction.topic_keywords);
        document.getElementById("topicName").textContent =
            `T${topicID}: ${keywords.join(", ")}`;
    }

    // ------------------- Topic Model Output --------------------
    const topicList = document.getElementById("topicsList");
    topicList.innerHTML = "";

    data.topics.forEach((topic, idx) => {
        const keywords = toArray(topic);
        const li = document.createElement("li");
        li.textContent = `T${idx}: ${keywords.join(", ")}`;
        li.className = "topic-item";
        topicList.appendChild(li);
    });

    // ------------------- Topic Bar Chart --------------------
    if (window.topicsChartRef) {
        window.topicsChartRef.destroy();
    }

    const ctx = document.getElementById("topicsChart").getContext("2d");
    window.topicsChartRef = new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.topics.map((_, i) => `T${i}`),
            datasets: [{
                label: "Topic weight",
                data: data.topics.map(() => 1),
                backgroundColor: "#4ea8de"
            }]
        },
        options: { responsive: true }
    });

    // ------------------- Summary --------------------
    document.getElementById("summaryText").textContent =
        data.summary || "No summary generated.";
    document.getElementById("summaryModeTag").textContent = data.summary_mode;

    // ------------------- Wordcloud --------------------
    const wcImg = document.getElementById("wordcloudImg");
    if (data.wordcloud) {
        wcImg.src = "data:image/png;base64," + data.wordcloud;
        wcImg.style.display = "block";
    } else {
        wcImg.style.display = "none";
    }

    // ------------------- Reports --------------------
    document.getElementById("reportJsonPath").innerHTML =
        `JSON: <a href="/${data.json_report_path}" target="_blank">${data.json_report_path}</a>`;

    document.getElementById("reportPdfPath").innerHTML =
        `PDF: <a href="/${data.pdf_report_path}" target="_blank">${data.pdf_report_path}</a>`;
}


// -------------------------------------------------
// ERROR + LOADING UI HELPERS
// -------------------------------------------------

function showError(msg) {
    const box = document.getElementById("errorMsg");
    box.textContent = msg;
    box.style.display = "block";

    setTimeout(() => {
        box.style.display = "none";
    }, 5000);
}

function loading(state) {
    const btn = document.getElementById("analyzeBtn");
    if (state) {
        btn.disabled = true;
        btn.textContent = "Analyzing...";
    } else {
        btn.disabled = false;
        btn.textContent = "Analyze Text";
    }
}
