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
        topics = topic_modeler.classify_topic(text)
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
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            file_result = {
                'filename': file.filename,
                'success': False,
                'error': None,
                'topics': None,
                'sentiment': None,
                'summary': None
            }
            
            try:
                # Extract text from file based on file type
                try:
                    text = extract_text_from_file(file, file.filename)
                except ImportError as e:
                    file_result['error'] = str(e)
                    results.append(file_result)
                    continue
                except Exception as e:
                    file_result['error'] = f'Error reading file: {str(e)}'
                    results.append(file_result)
                    continue
                
                if not text or len(text.strip()) == 0:
                    file_result['error'] = 'No text content found in the file'
                    results.append(file_result)
                    continue
                
                if len(text.strip()) < 50:
                    file_result['error'] = 'File content is too short. Please provide a file with at least 50 characters.'
                    results.append(file_result)
                    continue
                
                # Perform all analyses
                topics = topic_modeler.classify_topic(text)
                sentiment = sentiment_analyzer.analyze(text)
                summary = text_summarizer.summarize(text)
                
                file_result['topics'] = topics
                file_result['sentiment'] = sentiment
                file_result['summary'] = summary
                file_result['success'] = True
                
            except Exception as e:
                file_result['error'] = str(e)
            
            results.append(file_result)
        
        # Check if at least one file was processed successfully
        success_count = sum(1 for r in results if r.get('success', False))
        
        return jsonify({
            'results': results,
            'total_files': len(results),
            'successful_files': success_count,
            'success': success_count > 0
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)

