// Chart instances
let sentimentChart = null;
let wordFrequencyChart = null;
let trendChart = null;
let ldaChart = null;
let currentText = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeFileUpload();
    initializeAnalyzeButton();
    initializeLDA();
});

// Tab functionality
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            // Remove active class from all tabs and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            button.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

// File upload functionality
function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#6366f1';
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Handle file upload
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('textInput').value = e.target.result;
        // Switch to paste tab
        document.querySelector('[data-tab="paste"]').click();
    };
    reader.readAsText(file);
}

// Analyze button functionality
function initializeAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.addEventListener('click', () => {
        const text = getTextInput();
        if (!text || text.trim().length === 0) {
            alert('Please enter some text to analyze!');
            return;
        }
        analyzeText(text);
    });
}

// Get text input based on active tab
function getTextInput() {
    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');

    if (activeTab === 'paste') {
        return document.getElementById('textInput').value;
    } else if (activeTab === 'url') {
        const url = document.getElementById('urlInput').value;
        if (url) {
            // In a real application, you would fetch the URL content
            // For demo purposes, we'll show an alert
            alert('URL fetching would be implemented with a backend service. For now, please use the "Paste Text" tab.');
            return null;
        }
    }
    return null;
}

// Main analysis function
function analyzeText(text) {
    currentText = text;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');

    // Show loading state
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    // Simulate real-time processing
    setTimeout(() => {
        const analysis = performAnalysis(text);
        displayResults(analysis);

        // Reset button
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }, 500);
}

// Perform comprehensive text analysis
function performAnalysis(text) {
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

    return {
        text: text,
        wordCount: words.length,
        sentenceCount: sentences.length,
        characterCount: text.length,
        summary: generateSummary(text, sentences),
        sentiment: analyzeSentiment(text, sentences),
        topics: extractTopics(words, text),
        wordFrequency: calculateWordFrequency(words),
        trends: analyzeTrends(sentences),
        keyPhrases: extractKeyPhrases(text, sentences)
    };
}

// Generate summary
function generateSummary(text, sentences) {
    if (sentences.length <= 3) {
        return text;
    }

    // Simple extractive summarization - take first, middle, and last sentences
    const summarySentences = [
        sentences[0],
        sentences[Math.floor(sentences.length / 2)],
        sentences[sentences.length - 1]
    ];

    return summarySentences.join('. ') + '.';
}

// Sentiment analysis
function analyzeSentiment(text, sentences) {
    const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy', 'pleased', 'satisfied', 'best', 'brilliant', 'outstanding', 'perfect', 'positive', 'success', 'win', 'achievement'];
    const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'disappointed', 'worst', 'failure', 'problem', 'issue', 'error', 'wrong', 'negative', 'difficult', 'hard', 'struggle', 'pain'];

    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    let positiveCount = 0;
    let negativeCount = 0;

    words.forEach(word => {
        if (positiveWords.includes(word)) positiveCount++;
        if (negativeWords.includes(word)) negativeCount++;
    });

    const total = positiveCount + negativeCount;
    const positive = total > 0 ? (positiveCount / total) * 100 : 0;
    const negative = total > 0 ? (negativeCount / total) * 100 : 0;
    const neutral = total > 0 ? ((words.length - total) / words.length) * 100 : 100;

    let overall = 'neutral';
    let score = 0;

    if (positiveCount > negativeCount) {
        overall = 'positive';
        score = Math.min(100, 50 + (positiveCount - negativeCount) * 5);
    } else if (negativeCount > positiveCount) {
        overall = 'negative';
        score = Math.max(0, 50 - (negativeCount - positiveCount) * 5);
    } else {
        overall = 'neutral';
        score = 50;
    }

    return {
        overall,
        score,
        positive: Math.round(positive),
        negative: Math.round(negative),
        neutral: Math.round(neutral)
    };
}

// Extract topics and themes
function extractTopics(words, text) {
    // Common stop words to exclude
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how']);

    // Filter out stop words and short words
    const filteredWords = words.filter(word =>
        word.length > 3 && !stopWords.has(word)
    );

    // Count word frequency
    const wordCount = {};
    filteredWords.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1;
    });

    // Get top topics
    const topics = Object.entries(wordCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([word, count]) => ({
            word: word.charAt(0).toUpperCase() + word.slice(1),
            count,
            weight: Math.min(100, (count / filteredWords.length) * 100)
        }));

    return topics;
}

// Calculate word frequency
function calculateWordFrequency(words) {
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they']);

    const filteredWords = words.filter(word =>
        word.length > 3 && !stopWords.has(word)
    );

    const frequency = {};
    filteredWords.forEach(word => {
        frequency[word] = (frequency[word] || 0) + 1;
    });

    return Object.entries(frequency)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([word, count]) => ({
            word,
            count
        }));
}

// Analyze trends (sentiment over time)
function analyzeTrends(sentences) {
    const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy', 'pleased', 'satisfied', 'best', 'brilliant', 'outstanding', 'perfect', 'positive', 'success', 'win', 'achievement'];
    const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'disappointed', 'worst', 'failure', 'problem', 'issue', 'error', 'wrong', 'negative', 'difficult', 'hard', 'struggle', 'pain'];

    const segmentSize = Math.max(1, Math.floor(sentences.length / 5));
    const trends = [];

    for (let i = 0; i < sentences.length; i += segmentSize) {
        const segment = sentences.slice(i, i + segmentSize).join(' ');
        const words = segment.toLowerCase().match(/\b\w+\b/g) || [];

        let positive = 0;
        let negative = 0;

        words.forEach(word => {
            if (positiveWords.includes(word)) positive++;
            if (negativeWords.includes(word)) negative++;
        });

        const score = positive > negative ? 1 : negative > positive ? -1 : 0;
        trends.push({
            segment: Math.floor(i / segmentSize) + 1,
            score: score * 50 + 50, // Normalize to 0-100
            positive,
            negative
        });
    }

    return trends;
}

// Extract key phrases
function extractKeyPhrases(text, sentences) {
    const phrases = [];

    // Extract sentences with high information density (longer sentences)
    const importantSentences = sentences
        .filter(s => s.split(' ').length > 10)
        .slice(0, 5);

    importantSentences.forEach((sentence, index) => {
        phrases.push({
            text: sentence.trim().substring(0, 100) + (sentence.length > 100 ? '...' : ''),
            type: index === 0 ? 'Main Point' : index === 1 ? 'Supporting Detail' : 'Key Insight'
        });
    });

    return phrases;
}

// Display analysis results
function displayResults(analysis) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });

    // Update stats
    document.getElementById('wordCount').textContent = analysis.wordCount.toLocaleString();
    document.getElementById('sentenceCount').textContent = analysis.sentenceCount;
    document.getElementById('sentimentScore').textContent = analysis.sentiment.overall.toUpperCase();
    document.getElementById('topicCount').textContent = analysis.topics.length;

    // Update summary
    document.getElementById('summaryText').textContent = analysis.summary;

    // Display sentiment
    displaySentiment(analysis.sentiment);

    // Display topics
    displayTopics(analysis.topics);

    // Display word frequency
    displayWordFrequency(analysis.wordFrequency);

    // Display trends
    displayTrends(analysis.trends);

    // Display key phrases
    displayKeyPhrases(analysis.keyPhrases);
}

// Display sentiment analysis
function displaySentiment(sentiment) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');

    if (sentimentChart) {
        sentimentChart.destroy();
    }

    sentimentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [sentiment.positive, sentiment.neutral, sentiment.negative],
                backgroundColor: [
                    '#10b981',
                    '#f59e0b',
                    '#ef4444'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        padding: 15,
                        font: {
                            size: 14
                        }
                    }
                }
            }
        }
    });

    // Display sentiment details
    const detailsContainer = document.getElementById('sentimentDetails');
    detailsContainer.innerHTML = `
        <div class="sentiment-item">
            <div class="sentiment-label">Overall Sentiment</div>
            <div class="sentiment-value ${sentiment.overall}">${sentiment.overall.toUpperCase()}</div>
        </div>
        <div class="sentiment-item">
            <div class="sentiment-label">Sentiment Score</div>
            <div class="sentiment-value ${sentiment.overall}">${sentiment.score}%</div>
        </div>
        <div class="sentiment-item">
            <div class="sentiment-label">Positive</div>
            <div class="sentiment-value positive">${sentiment.positive}%</div>
        </div>
        <div class="sentiment-item">
            <div class="sentiment-label">Neutral</div>
            <div class="sentiment-value neutral">${sentiment.neutral}%</div>
        </div>
        <div class="sentiment-item">
            <div class="sentiment-label">Negative</div>
            <div class="sentiment-value negative">${sentiment.negative}%</div>
        </div>
    `;
}

// Display topics
function displayTopics(topics) {
    const container = document.getElementById('topicsContainer');
    container.innerHTML = topics.map(topic => `
        <div class="topic-tag">
            <span>${topic.word}</span>
            <span class="topic-weight">${Math.round(topic.weight)}%</span>
        </div>
    `).join('');
}

// Display word frequency
function displayWordFrequency(wordFreq) {
    const ctx = document.getElementById('wordFrequencyChart').getContext('2d');

    if (wordFrequencyChart) {
        wordFrequencyChart.destroy();
    }

    const labels = wordFreq.map(item => item.word);
    const data = wordFreq.map(item => item.count);

    wordFrequencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: data,
                backgroundColor: '#6366f1',
                borderColor: '#4f46e5',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#cbd5e1',
                        stepSize: 1
                    },
                    grid: {
                        color: '#475569'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#475569'
                    }
                }
            }
        }
    });
}

// Display trends
function displayTrends(trends) {
    const ctx = document.getElementById('trendChart').getContext('2d');

    if (trendChart) {
        trendChart.destroy();
    }

    const labels = trends.map(t => `Segment ${t.segment}`);
    const scores = trends.map(t => t.score);

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sentiment Trend',
                data: scores,
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointBackgroundColor: '#8b5cf6',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#cbd5e1',
                        callback: function(value) {
                            if (value < 50) return 'Negative';
                            if (value > 50) return 'Positive';
                            return 'Neutral';
                        }
                    },
                    grid: {
                        color: '#475569'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#475569'
                    }
                }
            }
        }
    });
}

// Display key phrases
function displayKeyPhrases(phrases) {
    const container = document.getElementById('phrasesContainer');

    if (phrases.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary);">No key phrases extracted.</p>';
        return;
    }

    container.innerHTML = phrases.map(phrase => `
        <div class="phrase-card">
            <div class="phrase-text">${phrase.text}</div>
            <div class="phrase-type">${phrase.type}</div>
        </div>
    `).join('');
}

// Initialize LDA functionality
function initializeLDA() {
    const runLDABtn = document.getElementById('runLDA');
    runLDABtn.addEventListener('click', () => {
        if (!currentText || currentText.trim().length === 0) {
            alert('Please analyze some text first!');
            return;
        }
        runLDAAnalysis();
    });
}

// Run LDA Analysis
function runLDAAnalysis() {
    const numTopics = parseInt(document.getElementById('numTopics').value) || 5;
    const numWords = parseInt(document.getElementById('numWords').value) || 10;
    const runLDABtn = document.getElementById('runLDA');
    const resultsContainer = document.getElementById('ldaResults');

    // Show loading
    runLDABtn.disabled = true;
    runLDABtn.textContent = '⏳ Processing LDA...';
    resultsContainer.innerHTML = '<p class="loading" style="color: var(--text-secondary); text-align: center; padding: 20px;">Running LDA topic modeling...</p>';

    // Run LDA in a timeout to prevent blocking
    setTimeout(() => {
        const ldaResults = performLDA(currentText, numTopics, numWords);
        displayLDAResults(ldaResults, numTopics);

        // Reset button
        runLDABtn.disabled = false;
        runLDABtn.textContent = 'Run LDA Analysis';
    }, 100);
}

// Perform LDA Topic Modeling
function performLDA(text, numTopics, numWords) {
    // Split text into documents (sentences)
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);

    if (sentences.length < numTopics) {
        // If not enough sentences, split into smaller chunks
        const chunkSize = Math.ceil(text.length / numTopics);
        const chunks = [];
        for (let i = 0; i < text.length; i += chunkSize) {
            chunks.push(text.substring(i, i + chunkSize));
        }
        return performLDACore(chunks, numTopics, numWords);
    }

    return performLDACore(sentences, numTopics, numWords);
}

// Core LDA Implementation (Simplified Gibbs Sampling)
function performLDACore(documents, numTopics, numWords) {
    const stopWords = new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
    ]);

    // Preprocess documents
    const processedDocs = documents.map(doc => {
        const words = doc.toLowerCase()
            .match(/\b\w+\b/g) || [];
        return words.filter(word =>
            word.length > 3 && !stopWords.has(word)
        );
    }).filter(doc => doc.length > 0);

    if (processedDocs.length === 0) {
        return [];
    }

    // Build vocabulary
    const vocabulary = new Set();
    processedDocs.forEach(doc => {
        doc.forEach(word => vocabulary.add(word));
    });
    const vocabArray = Array.from(vocabulary);
    const vocabIndex = {};
    vocabArray.forEach((word, idx) => {
        vocabIndex[word] = idx;
    });

    // Initialize topic assignments (simplified)
    const docTopicCounts = Array(processedDocs.length).fill(null).map(() => Array(numTopics).fill(0));
    const topicWordCounts = Array(numTopics).fill(null).map(() => Array(vocabArray.length).fill(0));
    const docTopicSums = Array(processedDocs.length).fill(0);
    const topicWordSums = Array(numTopics).fill(0);

    // Random initial assignment
    const assignments = [];
    processedDocs.forEach((doc, docIdx) => {
        const docAssignments = [];
        doc.forEach((word, wordIdx) => {
            const topic = Math.floor(Math.random() * numTopics);
            docAssignments.push(topic);
            const wordIdxInVocab = vocabIndex[word];
            docTopicCounts[docIdx][topic]++;
            topicWordCounts[topic][wordIdxInVocab]++;
            docTopicSums[docIdx]++;
            topicWordSums[topic]++;
        });
        assignments.push(docAssignments);
    });

    // Simplified Gibbs Sampling (fewer iterations for performance)
    const iterations = 20;
    const alpha = 0.1; // Dirichlet prior for document-topic
    const beta = 0.01; // Dirichlet prior for topic-word

    for (let iter = 0; iter < iterations; iter++) {
        processedDocs.forEach((doc, docIdx) => {
            doc.forEach((word, wordIdx) => {
                const currentTopic = assignments[docIdx][wordIdx];
                const wordIdxInVocab = vocabIndex[word];

                // Remove current assignment
                docTopicCounts[docIdx][currentTopic]--;
                topicWordCounts[currentTopic][wordIdxInVocab]--;
                docTopicSums[docIdx]--;
                topicWordSums[currentTopic]--;

                // Calculate probabilities for each topic
                const probs = [];
                let sum = 0;

                for (let topic = 0; topic < numTopics; topic++) {
                    const docTopicProb = (docTopicCounts[docIdx][topic] + alpha) / (docTopicSums[docIdx] + alpha * numTopics);
                    const wordTopicProb = (topicWordCounts[topic][wordIdxInVocab] + beta) / (topicWordSums[topic] + beta * vocabArray.length);
                    const prob = docTopicProb * wordTopicProb;
                    probs.push(prob);
                    sum += prob;
                }

                // Sample new topic
                let random = Math.random() * sum;
                let newTopic = 0;
                for (let i = 0; i < numTopics; i++) {
                    random -= probs[i];
                    if (random <= 0) {
                        newTopic = i;
                        break;
                    }
                }

                // Update assignment
                assignments[docIdx][wordIdx] = newTopic;
                docTopicCounts[docIdx][newTopic]++;
                topicWordCounts[newTopic][wordIdxInVocab]++;
                docTopicSums[docIdx]++;
                topicWordSums[newTopic]++;
            });
        });
    }

    // Extract top words for each topic
    const topics = [];
    for (let topic = 0; topic < numTopics; topic++) {
        const wordProbs = [];
        for (let wordIdx = 0; wordIdx < vocabArray.length; wordIdx++) {
            const prob = (topicWordCounts[topic][wordIdx] + beta) / (topicWordSums[topic] + beta * vocabArray.length);
            wordProbs.push({
                word: vocabArray[wordIdx],
                prob
            });
        }

        wordProbs.sort((a, b) => b.prob - a.prob);
        const topWords = wordProbs.slice(0, numWords);

        // Calculate topic distribution across documents
        let topicDocCount = 0;
        processedDocs.forEach((doc, docIdx) => {
            if (docTopicCounts[docIdx][topic] > 0) {
                topicDocCount++;
            }
        });
        const topicProbability = (topicDocCount / processedDocs.length) * 100;

        topics.push({
            id: topic + 1,
            words: topWords,
            probability: topicProbability
        });
    }

    // Sort topics by probability
    topics.sort((a, b) => b.probability - a.probability);

    return topics;
}

// Display LDA Results
function displayLDAResults(topics, numTopics) {
    const resultsContainer = document.getElementById('ldaResults');
    const chartContainer = document.getElementById('ldaChartContainer');

    if (topics.length === 0) {
        resultsContainer.innerHTML = '<p style="color: var(--text-secondary);">Unable to extract topics. Please try with more text.</p>';
        chartContainer.style.display = 'none';
        return;
    }

    // Display topics
    resultsContainer.innerHTML = topics.map(topic => `
        <div class="lda-topic">
            <div class="lda-topic-header">
                <div class="lda-topic-title">Topic ${topic.id}</div>
                <div class="lda-topic-probability">${topic.probability.toFixed(1)}%</div>
            </div>
            <div class="lda-topic-words">
                ${topic.words.map((w, idx) => `
                    <div class="lda-word">
                        <span>${w.word}</span>
                        <span class="lda-word-weight">${(w.prob * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');

    // Display topic distribution chart
    displayLDADistributionChart(topics);
}

// Display LDA Distribution Chart
function displayLDADistributionChart(topics) {
    const chartContainer = document.getElementById('ldaChartContainer');
    chartContainer.style.display = 'block';

    const ctx = document.getElementById('ldaChart').getContext('2d');

    if (ldaChart) {
        ldaChart.destroy();
    }

    const labels = topics.map(t => `Topic ${t.id}`);
    const data = topics.map(t => t.probability);
    const colors = [
        '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#3b82f6', '#a855f7', '#f97316', '#06b6d4', '#84cc16'
    ];

    ldaChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Topic Distribution (%)',
                data: data,
                backgroundColor: colors.slice(0, topics.length),
                borderColor: colors.slice(0, topics.length).map(c => c),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Distribution: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#cbd5e1',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: '#475569'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#475569'
                    }
                }
            }
        }
    });
}