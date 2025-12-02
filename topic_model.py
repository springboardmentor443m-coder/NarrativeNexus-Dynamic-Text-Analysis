"""
BERTopic Topic Modelling for AI-Narrative-Nexus
Loads pre-trained BERTopic model and provides inference
"""

import pickle
import os
from bertopic import BERTopic

CUSTOM_TOPIC_NAMES = {
    0:  "Narrative / story-style text",
    1:  "General explanations and statements",
    2:  "Cars and transportation discussions",
    3:  "Israel and Middle East politics",
    4:  "X Window System / GUI widgets",
    5:  "Cryptography and Clipper key debate",
    6:  "Space exploration and astronomy",
    7:  "Image density and P1–P3 graphics",
    8:  "Guns, weapons and gun control",
    9:  "Video cards and computer screens",
    10: "Basic descriptive sentences / misc text",
    11: "MS-DOS, Windows and disk files",
    12: "DOS software versions and quality reviews",
    13: "Personal experiences and anecdotes",
    14: "Operating systems and software versions (edu/comp)",
    15: "US federal budget and Clinton-era politics",
    16: "Waco siege fire and David Koresh",
    17: "Food, MSG and vitamin B6 health effects",
    18: "CPU speed, 68040 processor and FPU performance",
    19: "Sexuality and homosexuality discussions",
    20: "Disk drives, cables and storage hardware",
    21: "Stereo systems, speakers and audio channels",
    22: "Printers, fonts and TrueType printing",
    23: "SCSI / IDE disk controllers and interface chips",
    24: "Water, steam and heat transfer (thermodynamics)",
    25: "Computer memory (SIMMs, RAM capacity and access)",
    26: "Serial ports, modems and COM/IRQ settings",
    27: "Moral vs immoral behaviour and ethics",
    28: "Health risks of smokeless tobacco and E. coli",
    29: "Illegal drugs and substance abuse (cocaine, LSD)",
    -1: "Outlier / uncategorized documents",
}


class TopicPredictor:
    """
    Topic prediction using pre-trained BERTopic model.
    Loads saved model and provides inference on new documents.
    """

    def __init__(self):
        self.model = None
        self.topic_names = {}
        self.model_dir = os.path.dirname(__file__)
        self._load_model()

    def _load_model(self):
        """Load pre-trained BERTopic model and topic mapping."""
        try:
            model_path = os.path.join(self.model_dir, "bertopic_model")
            if not os.path.exists(model_path):
                print(f"⚠️  BERTopic model not found at {model_path}")
                print("Run train_bertopic.py first to train and save the model.")
                return False

            self.model = BERTopic.load(model_path)
            print(f"✅ Loaded BERTopic model from {model_path}")

            mapping_path = os.path.join(self.model_dir, "topic_mapping.pkl")
            if os.path.exists(mapping_path):
                with open(mapping_path, "rb") as f:
                    mapping = pickle.load(f)
                    self.topic_names = mapping.get("topic_names", {})
                print(f"✅ Loaded topic mapping with {len(self.topic_names)} topics")
            else:
                print("⚠️  topic_mapping.pkl not found in models directory")

            return True

        except Exception as e:
            print(f"❌ Error loading BERTopic model: {e}")
            return False

    def predict_topics(self, documents: list):
        """
        Predict topics for a list of documents.

        Args:
            documents: list[str]

        Returns:
            dict with predictions and topic info.
        """
        if not self.model:
            return {
                "status": "error",
                "error": "Model not loaded. Run train_bertopic.py first.",
            }

        try:
            topics, probs = self.model.transform(documents)

            result = {
                "status": "success",
                "n_documents": len(documents),
                "predictions": [],
            }

            for idx, (topic_id, prob) in enumerate(zip(topics, probs)):
                topic_name = CUSTOM_TOPIC_NAMES.get(
                    topic_id,
                    self.topic_names.get(topic_id, f"Topic_{topic_id}")
                )

                if topic_id == -1:
                    top_words = []
                else:
                    words_data = self.model.get_topic(topic_id)
                    top_words = [w for w, _ in words_data[:5]] if words_data else []

                result["predictions"].append(
                    {
                        "doc_idx": idx,
                        "doc_preview": documents[idx][:100],
                        "topic_id": int(topic_id),
                        "topic_name": topic_name,
                        "confidence": float(prob),
                        "top_keywords": top_words,
                    }
                )

            return result

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def predict_single(self, text: str):
        """
        Predict topic for a single document.

        Args:
            text: str

        Returns:
            dict with prediction.
        """
        if not self.model:
            return {
                "status": "error",
                "error": "Model not loaded. Run train_bertopic.py first.",
            }

        try:
            topics, probs = self.model.transform([text])
            topic_id = topics[0]
            prob = probs[0]

            topic_name = CUSTOM_TOPIC_NAMES.get(
                topic_id,
                self.topic_names.get(topic_id, f"Topic_{topic_id}")
            )

            if topic_id == -1:
                top_words = []
            else:
                words_data = self.model.get_topic(topic_id)
                top_words = [w for w, _ in words_data[:5]] if words_data else []

            return {
                "status": "success",
                "text_preview": text[:100],
                "topic_id": int(topic_id),
                "topic_name": topic_name,
                "confidence": float(prob),
                "top_keywords": top_words,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_topic_info(self):
        """Get information about all topics."""
        if not self.model:
            return {"status": "error", "error": "Model not loaded."}

        try:
            topics_info = {}
            for topic_id, auto_name in self.topic_names.items():
                name = CUSTOM_TOPIC_NAMES.get(topic_id, auto_name)

                if topic_id == -1:
                    topics_info[topic_id] = {
                        "name": name,
                        "keywords": [],
                    }
                else:
                    words_data = self.model.get_topic(topic_id)
                    keywords = [w for w, _ in words_data[:5]] if words_data else []
                    topics_info[topic_id] = {
                        "name": name,
                        "keywords": keywords,
                    }

            return {
                "status": "success",
                "n_topics": len([t for t in self.topic_names.keys() if t != -1]),
                "topics": topics_info,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


topic_predictor = TopicPredictor()
