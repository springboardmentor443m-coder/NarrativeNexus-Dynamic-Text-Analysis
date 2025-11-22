from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from typing import List, Dict, Tuple
from text_processor import TextProcessor


class LDAModeler:
    def _init_(self, num_topics: int = 5, passes: int = 10, alpha: str = "auto", beta: str = "auto"):
        self.num_topics = num_topics
        self.passes = passes
        self.alpha = alpha if alpha != "auto" else None
        self.beta = beta if beta != "auto" else None
        self.processor = TextProcessor()
        self.model = None
        self.dictionary = None
        self.corpus = None
    
    def prepare_corpus(self, texts: List[str]) -> Tuple[List[List[str]], corpora.Dictionary, List]:
        """Prepare corpus for LDA modeling"""
        # Process all texts
        processed_texts = [self.processor.process_for_lda(text) for text in texts]
        
        # Filter out empty texts
        processed_texts = [text for text in processed_texts if len(text) > 0]
        
        if len(processed_texts) == 0:
            raise ValueError("No valid texts to process for LDA")
        
        # Create dictionary
        dictionary = corpora.Dictionary(processed_texts)
        
        # Filter extremes
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        return processed_texts, dictionary, corpus
    
    def train(self, texts: List[str]) -> Dict:
        """Train LDA model on texts"""
        processed_texts, dictionary, corpus = self.prepare_corpus(texts)
        
        self.dictionary = dictionary
        self.corpus = corpus
        
        # Train LDA model
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            alpha=self.alpha,
            eta=self.beta,
            random_state=42,
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=self.model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        return {
            "model": self.model,
            "dictionary": self.dictionary,
            "corpus": self.corpus,
            "coherence_score": coherence_score,
            "num_topics": self.num_topics
        }
    
    def get_topics(self, num_words: int = 10) -> List[Dict]:
        """Get topics with their top words"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        topics = []
        for topic_id in range(self.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append({
                "topic_id": topic_id,
                "words": [{"word": word, "weight": weight} for word, weight in topic_words],
                "top_words": [word for word, _ in topic_words[:5]]
            })
        
        return topics
    
    def predict_topics(self, text: str) -> List[Dict]:
        """Predict topic distribution for a single text"""
        if self.model is None or self.dictionary is None:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.processor.process_for_lda(text)
        bow = self.dictionary.doc2bow(processed_text)
        
        # Get topic distribution
        topic_dist = self.model[bow]
        
        # Format results
        topics = []
        for topic_id, prob in topic_dist:
            topic_words = self.model.show_topic(topic_id, topn=5)
            topics.append({
                "topic_id": int(topic_id),
                "probability": float(prob),
                "top_words": [word for word, _ in topic_words]
            })
        
        # Sort by probability
        topics.sort(key=lambda x: x["probability"], reverse=True)
        
        return topics
    
    def serialize_model(self) -> Dict:
        """Serialize model for storage"""
        if self.model is None or self.dictionary is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Serialize dictionary
        dictionary_data = {
            "id2token": {id: token for id, token in self.dictionary.id2token.items()},
            "token2id": self.dictionary.token2id
        }
        
        # Serialize model
        model_data = {
            "num_topics": self.model.num_topics,
            "alpha": self.model.alpha if isinstance(self.model.alpha, (int, float)) else list(self.model.alpha),
            "eta": self.model.eta if isinstance(self.model.eta, (int, float)) else list(self.model.eta),
            "topics": self.get_topics()
        }
        
        return {
            "dictionary": dictionary_data,
            "model": model_data,
            "num_topics": self.num_topics,
            "passes": self.passes
        }
    
    def load_model(self, model_data: Dict):
        """Load model from serialized data"""
        # Note: Full model reconstruction would require more complex serialization
        # For now, we'll store the model parameters and topics
        # Dictionary data is stored but not fully reconstructed here
        # In production, you'd want to fully serialize/deserialize the Gensim model
        
        self.num_topics = model_data["num_topics"]
        self.passes = model_data.get("passes", 10)
        
        return model_data
