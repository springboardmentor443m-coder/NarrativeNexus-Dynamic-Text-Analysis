from flask import Flask, render_template, request, jsonify
import os
from config import Config
from models.topic_modeling import TopicModeler
from models.sentiment_analyzer import SentimentAnalyzer
from models.summarizer import TextSummarizer
from utils.file_parser import extract_text_from_file

app = Flask(__name__)
app.config.from_object(Config)

# Initialize models
topic_modeler = TopicModeler()
sentiment_analyzer = SentimentAnalyzer()
text_summarizer = TextSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Perform all analyses
        topics = topic_modeler.get_topics(text)
        sentiment = sentiment_analyzer.analyze(text)
        summary = text_summarizer.summarize(text)
        
        return jsonify({
            'topics': topics,
            'sentiment': sentiment,
            'summary': summary,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Extract text from file based on file type
        try:
            text = extract_text_from_file(file, file.filename)
        except ImportError as e:
            return jsonify({'error': str(e), 'success': False}), 500
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}', 'success': False}), 400
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'No text content found in the file', 'success': False}), 400
        
        if len(text.strip()) < 50:
            return jsonify({'error': 'File content is too short. Please provide a file with at least 50 characters.', 'success': False}), 400
        
        # Perform all analyses
        topics = topic_modeler.get_topics(text)
        sentiment = sentiment_analyzer.analyze(text)
        summary = text_summarizer.summarize(text)
        
        return jsonify({
            'topics': topics,
            'sentiment': sentiment,
            'summary': summary,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)

