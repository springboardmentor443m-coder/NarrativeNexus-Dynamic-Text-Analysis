import os
import json
import numpy as np
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer

class TopicModeler:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the TopicModeler by loading pre-computed centroids and metadata.
        Optimized to pre-calculate centroid norms for faster inference.
        """
        # Define paths relative to this file
        self.base_dir = os.path.join(os.path.dirname(__file__), "models", "bertopic_20newsgroups")
        self.centroids_path = os.path.join(self.base_dir, "topic_centroids.npy")
        self.map_path = os.path.join(self.base_dir, "topic_name_mapping.json")
        
        # Validate and Load Assets
        self._validate_assets()
        self.centroids = np.load(self.centroids_path)
        
        # Optimization: Pre-calculate norms for faster cosine similarity
        # This avoids re-calculating static vector magnitudes on every request
        self.centroid_norms = np.linalg.norm(self.centroids, axis=1)
        
        with open(self.map_path, "r", encoding="utf-8") as f:
            self.topic_names = json.load(f)

        # Load Encoder Model
        # Using CPU by default for broad compatibility; change to "cuda" if GPU available
        self.model = SentenceTransformer(embed_model_name, device="cpu")

    def _validate_assets(self):
        """Ensures all required model files exist before loading."""
        required_files = [self.centroids_path, self.map_path]
        for fpath in required_files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing model asset: {fpath}")

    def _cosine_similarity(self, vec: np.ndarray) -> np.ndarray:
        """Calculates cosine similarity between input vector and all topic centroids."""
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return np.zeros(len(self.centroids))
        
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        dot_products = np.dot(self.centroids, vec)
        return dot_products / (self.centroid_norms * vec_norm)

    def find_closest_topic(self, text: str, threshold: float = 0.35) -> Tuple[Optional[int], Optional[str], float]:
        """
        Embeds text and finds the semantically closest topic.
        Returns: (Topic ID, Topic Name, Similarity Score)
        """
        if not text or not text.strip():
             return None, None, 0.0

        try:
            # 1. Generate Embedding
            text_embedding = self.model.encode(text, convert_to_numpy=True)
            
            # 2. Calculate Similarities
            similarities = self._cosine_similarity(text_embedding)
            
            # 3. Find Best Match
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            # 4. Threshold Check
            if best_score < threshold:
                return None, None, 0.0
                
            # 5. Resolve Name (JSON keys are strings)
            topic_name = self.topic_names.get(str(best_idx), f"Topic {best_idx}")
            
            return best_idx, topic_name, best_score

        except Exception as e:
            print(f"[TopicModeler] Error analyzing text: {e}")
            return None, None, 0.0