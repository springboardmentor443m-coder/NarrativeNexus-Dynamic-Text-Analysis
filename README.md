# NarrativeNexus – Dynamic Text Analysis (Transformer-Powered)

A modern, scalable **Streamlit-based NLP application** for deep text analysis using **Transformer models + Hybrid LDA**.  
This system supports **large documents (50–200MB)** with optimized performance for:

-  Summarization (BART CNN)
-  Sentiment Analysis (RoBERTa – chunk-safe for long text)
-  Topic Modeling (Hybrid LDA for large documents)
-  Key Themes Extraction
-  Word Cloud Visualization
-  PDF, TXT, CSV, DOCX Upload Support

---

##  Features

- **Large File Support** (50–200MB safely)
- **Chunked NLP Processing** (prevents GPU/CPU memory crashes)
- **Hybrid Topic Modeling** using TF-IDF + LDA
- **High Accuracy Sentiment Analysis** with `twitter-roberta-base`
- **Adaptive Summarization** with word-target control
- **Modern Dark UI with Glow Cards**
- **Theme Switching via Sidebar**
- **Interactive Visuals using Plotly & Matplotlib**

---

##  Models Used

| Feature | Model |
|--------|--------|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Summary | `facebook/bart-large-cnn` |
| Tokenization | NLTK Punkt |
| Topic Modeling | Hybrid LDA (Scikit-Learn) |
| Visualization | WordCloud, Matplotlib, Plotly |

---

##  Installation (Windows)

### 1️ Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
