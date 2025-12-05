from transformers import pipeline
import torch

class ContentCondenser:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.condenser_pipeline = None
        self._initialize_condenser()
    
    def _initialize_condenser(self):
        """Load BART model for content condensation"""
        try:
            self.condenser_pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Error loading BART model: {e}")
            print("Falling back to extractive condensation...")
            self.condenser_pipeline = None
    
    def condense_content(self, text, max_length=150, min_length=50, ratio=0.3):
        """Condense text content using BART model"""
        if not text or len(text.strip()) == 0:
            return {
                'summary': '',
                'original_length': 0,
                'summary_length': 0,
                'compression_ratio': 0
            }
        
        original_length = len(text.split())
        
        try:
            if self.condenser_pipeline:
                # Calculate appropriate lengths based on input
                if original_length < 50:
                    max_length = min(original_length, 30)
                    min_length = max(5, original_length // 3)
                else:
                    max_length = min(max_length, original_length // 2)
                    min_length = min(min_length, original_length // 4)
                
                condensed = self.condenser_pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                
                condensed_length = len(condensed.split())
                compression_ratio = round(condensed_length / original_length, 2) if original_length > 0 else 0
                
                return {
                    'summary': condensed,
                    'original_length': original_length,
                    'summary_length': condensed_length,
                    'compression_ratio': compression_ratio
                }
            else:
                # Fallback to extractive condensation
                return self._extractive_condense(text, ratio)
        
        except Exception as e:
            print(f"Error in content condensation: {e}")
            return self._extractive_condense(text, ratio)
    
    def _extractive_condense(self, text, ratio=0.3):
        """Extractive condensation as fallback"""
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return {
                'summary': text[:200] + '...' if len(text) > 200 else text,
                'original_length': len(text.split()),
                'summary_length': len(text.split()),
                'compression_ratio': 1.0
            }
        
        # Calculate word frequencies
        stop_words = set(stopwords.words('english'))
        word_freq = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word not in stop_words and word.isalnum():
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words if word.isalnum())
            sentence_scores[sentence] = score
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        condensed_sentences = [sent for sent, _ in sorted(top_sentences, key=lambda x: sentences.index(x[0]))]
        
        condensed = ' '.join(condensed_sentences)
        original_length = len(text.split())
        condensed_length = len(condensed.split())
        compression_ratio = round(condensed_length / original_length, 2) if original_length > 0 else 0
        
        return {
            'summary': condensed,
            'original_length': original_length,
            'summary_length': condensed_length,
            'compression_ratio': compression_ratio
        }
