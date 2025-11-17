// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Analyze text input
async function analyzeText() {
    const text = document.getElementById('text-input').value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (text.length < 50) {
        showError('Please enter at least 50 characters for meaningful analysis.');
        return;
    }
    
    const loader = document.getElementById('loader');
    loader.style.display = 'block';
    hideResults();
    hideError();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during analysis.');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loader.style.display = 'none';
    }
}

// Analyze file upload
async function analyzeFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file to analyze.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const loader = document.getElementById('file-loader');
    loader.style.display = 'block';
    hideResults();
    hideError();
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during analysis.');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loader.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    
    // Display topics
    displayTopics(data.topics);
    
    // Display sentiment
    displaySentiment(data.sentiment);
    
    // Display summary
    displaySummary(data.summary);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display topics
function displayTopics(topics) {
    const topicsResult = document.getElementById('topics-result');
    
    if (!topics || !topics.topics || topics.topics.length === 0) {
        topicsResult.innerHTML = '<div class="empty-state"><p>' + 
            (topics.message || 'No topics found.') + '</p></div>';
        return;
    }
    
    let html = '';
    topics.topics.forEach(topic => {
        html += `
            <div class="topic-item">
                <div class="topic-header">
                    <span class="topic-id">Topic ${topic.topic_id}</span>
                    <span class="topic-weight">Weight: ${topic.weight}</span>
                </div>
                <div class="topic-words">
                    ${topic.words.map(word => `<span class="word-tag">${word}</span>`).join('')}
                </div>
            </div>
        `;
    });
    
    topicsResult.innerHTML = html;
}

// Display sentiment
function displaySentiment(sentiment) {
    const sentimentResult = document.getElementById('sentiment-result');
    
    if (!sentiment) {
        sentimentResult.innerHTML = '<div class="empty-state"><p>Sentiment analysis unavailable.</p></div>';
        return;
    }
    
    const sentimentClass = `sentiment-${sentiment.sentiment}`;
    const confidence = sentiment.confidence || (sentiment.score * 100).toFixed(2);
    
    const html = `
        <div class="sentiment-display">
            <span class="sentiment-badge ${sentimentClass}">
                ${sentiment.sentiment.toUpperCase()}
            </span>
            <div class="sentiment-details">
                <p><strong>Label:</strong> ${sentiment.label}</p>
                <p><strong>Score:</strong> ${sentiment.score}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
                <p style="margin-top: 5px;"><strong>Confidence:</strong> ${confidence}%</p>
            </div>
        </div>
    `;
    
    sentimentResult.innerHTML = html;
}

// Display summary
function displaySummary(summary) {
    const summaryResult = document.getElementById('summary-result');
    
    if (!summary || !summary.summary) {
        summaryResult.innerHTML = '<div class="empty-state"><p>Summary unavailable.</p></div>';
        return;
    }
    
    const html = `
        <div class="summary-text">${summary.summary}</div>
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-label">Original Length</div>
                <div class="stat-value">${summary.original_length} words</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Summary Length</div>
                <div class="stat-value">${summary.summary_length} words</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Compression Ratio</div>
                <div class="stat-value">${(summary.compression_ratio * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
    
    summaryResult.innerHTML = html;
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Hide error message
function hideError() {
    document.getElementById('error-message').style.display = 'none';
}

// Hide results
function hideResults() {
    document.getElementById('results-section').style.display = 'none';
}

// File input change handler
document.getElementById('file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const label = document.querySelector('.file-label span');
        label.textContent = `Selected: ${file.name}`;
    }
});

// Allow drag and drop for files
const fileLabel = document.querySelector('.file-label');
const fileInput = document.getElementById('file-input');

fileLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileLabel.style.background = '#e8ebff';
});

fileLabel.addEventListener('dragleave', () => {
    fileLabel.style.background = '#f8f9ff';
});

fileLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    fileLabel.style.background = '#f8f9ff';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        const label = document.querySelector('.file-label span');
        label.textContent = `Selected: ${files[0].name}`;
    }
});

