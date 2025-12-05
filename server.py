from flask import Flask, render_template, request, jsonify
import os
from config import Config
from analyzers.theme_extractor import ThemeExtractor
from analyzers.emotion_detector import EmotionDetector
from analyzers.content_condenser import ContentCondenser
from utils.file_parser import extract_text_from_file

web_app = Flask(__name__, template_folder='views')
web_app.config.from_object(Config)

# Initialize analyzers
theme_extractor = ThemeExtractor()
emotion_detector = EmotionDetector()
content_condenser = ContentCondenser()

@web_app.route('/')
def home_page():
    return render_template('index.html')

@web_app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        
        if not input_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Perform all analyses
        themes = theme_extractor.extract_themes(input_text)
        emotion = emotion_detector.detect_emotion(input_text)
        condensed = content_condenser.condense_content(input_text)
        
        return jsonify({
            'topics': themes,
            'sentiment': emotion,
            'summary': condensed,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@web_app.route('/process-file', methods=['POST'])
def process_file():
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
                themes = theme_extractor.extract_themes(text)
                emotion = emotion_detector.detect_emotion(text)
                condensed = content_condenser.condense_content(text)
                
                file_result['topics'] = themes
                file_result['sentiment'] = emotion
                file_result['summary'] = condensed
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
    os.makedirs(web_app.config['UPLOAD_FOLDER'], exist_ok=True)
    web_app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
