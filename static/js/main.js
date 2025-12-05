// Mode switching functionality
function changeMode(mode) {
  // Hide all mode panels
  document.querySelectorAll(".mode-panel").forEach((panel) => {
    panel.classList.remove("active");
  });

  // Remove active class from all buttons
  document.querySelectorAll(".mode-button").forEach((btn) => {
    btn.classList.remove("active");
  });

  // Show selected mode panel
  document.getElementById(mode + "-mode").classList.add("active");

  // Add active class to clicked button
  event.target.classList.add("active");
}

// Process text content
async function processContent() {
  const content = document.getElementById("content-input").value.trim();

  if (!content) {
    showAlert("Please enter some content to process.");
    return;
  }

  if (content.length < 50) {
    showAlert("Please enter at least 50 characters for meaningful analysis.");
    return;
  }

  const spinner = document.getElementById("text-spinner");
  spinner.style.display = "block";
  hideOutput();
  hideAlert();

  try {
    const response = await fetch("/process", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: content }),
    });

    const data = await response.json();

    if (data.success) {
      renderOutput(data);
    } else {
      showAlert(data.error || "An error occurred during processing.");
    }
  } catch (error) {
    showAlert("Network error: " + error.message);
  } finally {
    spinner.style.display = "none";
  }
}

// Process document upload
async function processDocument() {
  const fileInput = document.getElementById("document-input");
  const file = fileInput.files[0];

  if (!file) {
    showAlert("Please select a document to process.");
    return;
  }

  const formData = new FormData();
  formData.append("files", file);

  const spinner = document.getElementById("file-spinner");
  spinner.style.display = "block";
  hideOutput();
  hideAlert();

  try {
    const response = await fetch("/process-file", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success && data.results && data.results.length > 0) {
      const firstResult = data.results[0];
      renderOutput({
        topics: firstResult.topics,
        sentiment: firstResult.sentiment,
        summary: firstResult.summary,
        success: true,
      });
    } else {
      showAlert(data.error || "An error occurred during processing.");
    }
  } catch (error) {
    showAlert("Network error: " + error.message);
  } finally {
    spinner.style.display = "none";
  }
}

// Render analysis output
function renderOutput(data) {
  const outputArea = document.getElementById("output-area");
  outputArea.style.display = "block";

  // Display themes
  renderThemes(data.topics);

  // Display emotion
  renderEmotion(data.sentiment);

  // Display condensation
  renderCondensation(data.summary);

  // Scroll to output
  outputArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Render theme extraction results
function renderThemes(themes) {
  const themeOutput = document.getElementById("theme-output");

  if (!themes || !themes.all_topics || themes.all_topics.length === 0) {
    const message = themes?.message || themes?.error || "No themes found.";
    themeOutput.innerHTML =
      '<div class="empty-message"><p>' + message + "</p></div>";
    return;
  }

  let html = "";
  if (themes.all_topics && themes.all_topics.length > 0) {
    themes.all_topics.forEach((theme) => {
      html += `
                <div class="theme-entry">
                    <div class="theme-title-bar">
                        <span class="theme-number">Theme ${
                          theme.topic_id
                        }</span>
                        <span class="theme-score">Score: ${theme.score}</span>
                    </div>
                    <div class="theme-keywords">
                        ${(theme.top_words || [])
                          .map(
                            (word) =>
                              `<span class="keyword-badge">${word}</span>`
                          )
                          .join("")}
                    </div>
                </div>
            `;
    });
  }

  themeOutput.innerHTML =
    html || '<div class="empty-message"><p>No themes found.</p></div>';
}

// Render emotion detection results
function renderEmotion(emotion) {
  const emotionOutput = document.getElementById("emotion-output");

  if (!emotion) {
    emotionOutput.innerHTML =
      '<div class="empty-message"><p>Emotion detection unavailable.</p></div>';
    return;
  }

  const emotionClass = `emotion-${emotion.sentiment}`;
  const confidence = emotion.confidence || (emotion.score * 100).toFixed(2);

  const html = `
        <div class="emotion-container">
            <span class="emotion-indicator ${emotionClass}">
                ${emotion.sentiment.toUpperCase()}
            </span>
            <div class="emotion-info">
                <p><strong>Label:</strong> ${emotion.label}</p>
                <p><strong>Score:</strong> ${emotion.score}</p>
                <div class="accuracy-meter">
                    <div class="accuracy-fill" style="width: ${confidence}%"></div>
                </div>
                <p style="margin-top: 8px;"><strong>Confidence:</strong> ${confidence}%</p>
            </div>
        </div>
    `;

  emotionOutput.innerHTML = html;
}

// Render content condensation results
function renderCondensation(condensation) {
  const condensationOutput = document.getElementById("condensation-output");

  if (!condensation || !condensation.summary) {
    condensationOutput.innerHTML =
      '<div class="empty-message"><p>Content condensation unavailable.</p></div>';
    return;
  }

  const html = `
        <div class="condensed-text">${condensation.summary}</div>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Original Length</div>
                <div class="metric-value">${
                  condensation.original_length
                } words</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Condensed Length</div>
                <div class="metric-value">${
                  condensation.summary_length
                } words</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Compression Ratio</div>
                <div class="metric-value">${(
                  condensation.compression_ratio * 100
                ).toFixed(1)}%</div>
            </div>
        </div>
    `;

  condensationOutput.innerHTML = html;
}

// Show alert message
function showAlert(message) {
  const alertBox = document.getElementById("alert-box");
  alertBox.textContent = message;
  alertBox.style.display = "block";
  alertBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Hide alert message
function hideAlert() {
  document.getElementById("alert-box").style.display = "none";
}

// Hide output area
function hideOutput() {
  document.getElementById("output-area").style.display = "none";
}

// File input change handler
document
  .getElementById("document-input")
  .addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      const label = document.querySelector(".upload-label span");
      label.textContent = `Selected: ${file.name}`;
    }
  });

// Allow drag and drop for files
const uploadLabel = document.querySelector(".upload-label");
const documentInput = document.getElementById("document-input");

uploadLabel.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadLabel.style.background = "#d0e7ff";
});

uploadLabel.addEventListener("dragleave", () => {
  uploadLabel.style.background = "#f0f7ff";
});

uploadLabel.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadLabel.style.background = "#f0f7ff";
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    documentInput.files = files;
    const label = document.querySelector(".upload-label span");
    label.textContent = `Selected: ${files[0].name}`;
  }
});
