import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from dotenv import load_dotenv
from groq import Groq

# --- Configuration ---
# Load environment variables for Groq API
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Ensure paths are absolute/relative safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bertopic_20newsgroups")
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_TOPICS = 40  # Reduce 20 Newsgroups to ~40 distinct themes

def load_dataset():
    """Fetches the 20 Newsgroups dataset for training."""
    print(f"[1/7] Loading dataset...")
    # Removing headers/footers ensures the model learns content, not email metadata
    data = fetch_20newsgroups(subset='all', remove=("headers", "footers", "quotes"))
    print(f"      Loaded {len(data.data)} documents.")
    return data.data

def build_pipeline():
    """Configures the modular BERTopic pipeline."""
    print(f"[2/7] Configuring model pipeline ({EMBED_MODEL})...")
    
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # UMAP: Dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric="cosine", 
        random_state=42  # Fixed seed for reproducibility
    )

    # HDBSCAN: Clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=25, 
        min_samples=5, 
        metric="euclidean", 
        cluster_selection_method="eom", 
        prediction_data=True
    )

    return embedder, umap_model, hdbscan_model

def train_model(embedder, umap_model, hdbscan_model, docs):
    """Trains and reduces the topic model."""
    print(f"[3/7] Training BERTopic (this may take time)...")
    
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    initial_count = len(topic_model.get_topic_info()) - 1 # Exclude -1
    print(f"      Initial topics found: {initial_count}")
    
    # Reduce topics to manageability
    if initial_count > TARGET_TOPICS:
        print(f"      Reducing to {TARGET_TOPICS} topics...")
        topic_model.reduce_topics(docs, nr_topics=TARGET_TOPICS)
    
    return topic_model

def generate_smart_labels(topic_model):
    """
    Uses Groq LLM to generate human-readable labels for each topic.
    Falls back to default labels if API key is missing or error occurs.
    """
    print(f"[4/7] Generating AI Topic Labels (using Groq)...")
    
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found. Using default topic names.")
        return None

    client = Groq(api_key=GROQ_API_KEY)
    topic_info = topic_model.get_topic_info()
    mapping = {}
    
    # Filter out outlier topic (-1) initially, handle it separately
    valid_topics = topic_info[topic_info['Topic'] != -1]
    
    # Outlier label is static
    mapping["-1"] = "General / Uncategorized"

    print(f"Labeling {len(valid_topics)} topics...")

    for index, row in valid_topics.iterrows():
        topic_id = row['Topic']
        keywords = row['Representation'][:10] # Top 10 keywords
        # Get representative docs (if available, take the first one truncated)
        rep_docs = topic_model.get_representative_docs(topic_id)
        doc_snippet = rep_docs[0][:500] if rep_docs else "No representative document."

        prompt = (
            f"Analyze the following Topic keywords and document snippet.\n"
            f"Keywords: {keywords}\n"
            f"Document: {doc_snippet}\n\n"
            f"Provide a short, professional, descriptive category label (3-5 words max) for this topic.\n"
            f"Examples: 'Space & Astronomy', 'Automotive Issues', 'Middle East Politics'.\n"
            f"Return ONLY the label as a raw string. Do not include 'Topic' or quotes."
        )

        try:
            # Rate limit handling (simple sleep)
            time.sleep(0.5) 
            
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            label = completion.choices[0].message.content.strip().replace('"', '')
            mapping[str(topic_id)] = label
            print(f"      Topic {topic_id}: {label}")
            
        except Exception as e:
            print(f"      ❌ Error labeling Topic {topic_id}: {e}")
            # Fallback to default name
            mapping[str(topic_id)] = row['Name']

    return mapping

def save_centroids(topic_model, embedder):
    """Calculates and saves centroid vectors."""
    print(f"[6/7] Calculating and saving topic centroids...")
    
    topic_info = topic_model.get_topic_info()
    valid_topics = sorted(topic_info[topic_info['Topic'] != -1]['Topic'].tolist())
    
    if not valid_topics:
        return

    max_id = max(valid_topics)
    embedding_dim = 384 
    centroids = np.zeros((max_id + 1, embedding_dim))
    
    for t_id in valid_topics:
        repr_docs = topic_model.get_representative_docs(t_id)
        if repr_docs:
            embeddings = embedder.encode(repr_docs)
            centroid = np.mean(embeddings, axis=0)
            centroids[t_id] = centroid
            
    np.save(os.path.join(MODEL_DIR, "topic_centroids.npy"), centroids)

def save_artifacts(topic_model, custom_mapping):
    """Saves the model, metadata, and the AI-generated name mapping."""
    print(f"[7/7] Saving model artifacts...")
    
    # 1. Main Model
    topic_model.save(os.path.join(MODEL_DIR, "trained_bertopic.pkl"))
    
    # 2. Metadata
    topic_info = topic_model.get_topic_info()
    topic_info.to_json(os.path.join(MODEL_DIR, "topic_info.json"), orient="records")
    
    # 3. Name Mapping
    # Use custom AI mapping if available, otherwise default
    if custom_mapping:
        final_mapping = custom_mapping
    else:
        # Fallback: Convert int64 keys to strings
        final_mapping = {str(row['Topic']): row['Name'] for _, row in topic_info.iterrows()}
    
    with open(os.path.join(MODEL_DIR, "topic_name_mapping.json"), "w") as f:
        json.dump(final_mapping, f, indent=4)
        
    print(f"\n✅ Success! AI-Labeled Model assets saved to:\n   {MODEL_DIR}")

if __name__ == "__main__":
    # Pipeline Execution
    documents = load_dataset()
    embedder, umap, hdbscan = build_pipeline()
    
    trained_model = train_model(embedder, umap, hdbscan, documents)
    
    # Generate AI labels using Groq
    ai_mapping = generate_smart_labels(trained_model)
    
    save_centroids(trained_model, embedder)
    save_artifacts(trained_model, ai_mapping)