import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from unidecode import unidecode
from collections import Counter
import os
from fastapi import HTTPException
import csv

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """Comprehensive text preprocessing for NLP tasks"""
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        soup = BeautifulSoup(text, 'html.parser')
        for script in soup(['script', 'style', 'head', 'title', 'meta']):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    
    def fix_encoding(self, text: str) -> str:
        """Fix encoding issues like Â, Ã, €"""
        text = unidecode(text)
        replacements = {
            'Â': '',
            'Ã©': 'e',
            'Ã¨': 'e',
            'Ã ': 'a',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            '...': '.',
            '…': '.'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)
        return text
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        return re.sub(r'\S+@\S+', '', text)
    
    def remove_social_media_artifacts(self, text: str) -> str:
        """Remove hashtags, mentions"""
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters, keep letters & punctuation"""
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_duplicates(self, text: str) -> str:
        """Remove duplicate sentences"""
        sentences = sent_tokenize(text)
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            normalized = sentence.lower().strip()
            if normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_sentences.append(sentence)
        return ' '.join(unique_sentences)
    
    def to_lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_stopwords(self, text: str) -> str:
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered)
    
    def lemmatize(self, text: str) -> str:
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized)
    
    def tokenize(self, text: str) -> list:
        return word_tokenize(text)
    
    def get_statistics(self, original_text: str, processed_text: str) -> dict:
        original_tokens = word_tokenize(original_text)
        processed_tokens = word_tokenize(processed_text)
        return {
            'original_char_count': len(original_text),
            'processed_char_count': len(processed_text),
            'original_word_count': len(original_tokens),
            'processed_word_count': len(processed_tokens),
            'unique_words': len(set(processed_tokens)),
            'sentences': len(sent_tokenize(processed_text)),
            'reduction_percentage': round((1 - len(processed_text)/len(original_text)) * 100, 2) if len(original_text) > 0 else 0
        }
    
    def preprocess(self, text: str,
                   remove_html_tags=True,
                   fix_encoding_issues=True,
                   remove_urls_emails=True,
                   remove_social=True,
                   remove_special_chars=True,
                   remove_duplicate_sentences=True,
                   lowercase=True,
                   remove_stops=True,
                   lemmatize_text=True) -> dict:
        original_text = text
        steps_applied = []
        if remove_html_tags:
            text = self.remove_html(text)
            steps_applied.append('HTML removal')
        if fix_encoding_issues:
            text = self.fix_encoding(text)
            steps_applied.append('Encoding fixes')
        if remove_urls_emails:
            text = self.remove_urls(text)
            text = self.remove_emails(text)
            steps_applied.append('URL/Email removal')
        if remove_social:
            text = self.remove_social_media_artifacts(text)
            steps_applied.append('Social media cleanup')
        if remove_duplicate_sentences:
            text = self.remove_duplicates(text)
            steps_applied.append('Duplicate removal')
        if remove_special_chars:
            text = self.remove_special_characters(text)
            steps_applied.append('Special character removal')
        if lowercase:
            text = self.to_lowercase(text)
            steps_applied.append('Lowercasing')
        if remove_stops:
            text = self.remove_stopwords(text)
            steps_applied.append('Stopword removal')
        if lemmatize_text:
            text = self.lemmatize(text)
            steps_applied.append('Lemmatization')
        stats = self.get_statistics(original_text, text)
        tokens = word_tokenize(text)
        top_words = Counter(tokens).most_common(20)
        return {
            'original_text': original_text[:500] + '...' if len(original_text) > 500 else original_text,
            'processed_text': text,
            'statistics': stats,
            'steps_applied': steps_applied,
            'top_words': top_words,
            'sample_tokens': tokens[:50]
        }


def extract_text_from_file(file_path: str, extension: str) -> str:
    """Extract text from TXT or CSV files."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif extension == '.csv':
            text_parts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'content' in row and row['content'].strip():
                        text_parts.append(row['content'].strip())
                    else:
                        values = list(row.values())
                        if len(values) > 1:
                            text_parts.append(' '.join(values[1:]).strip())
                        elif len(values) == 1 and values[0].strip():
                            text_parts.append(values[0].strip())
            if not text_parts:
                raise HTTPException(status_code=400, detail="No content found in CSV")
            return ' '.join(text_parts)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")









