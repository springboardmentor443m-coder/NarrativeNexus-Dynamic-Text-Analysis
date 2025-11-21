Dynamic Text Analysis - Frontend (lightweight)
============================================

Files included:
- index.html
- styles.css
- app.js
- README.md

How to use:
1. Unzip the archive and open index.html in your browser.
2. Paste text or upload a .txt/.csv file. (.docx parsing not included).
3. Click Analyze.
4. The app will try to POST to http://localhost:8000/analyze if 'Local API :8000' is selected.
   Expected backend JSON response (example):
     {
       "summary": "Short summary text",
       "sentiment": {"polarity": 0, "label":"neutral"},
       "topic": "topic words..."
     }
   If the backend is not available, the app falls back to a client-side summarizer.

Feel free to modify app.js to integrate with your actual backend endpoints.
