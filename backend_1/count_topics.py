import numpy as np

# Load labels assigned by HDBSCAN during training
labels = np.load("backend/models/topic_model/topic_labels.npy")

# Unique topic IDs (excluding noise label -1)
unique_topics = set(labels)
if -1 in unique_topics:
    unique_topics.remove(-1)  # -1 means "noise" or "unclustered"

print("Number of topics discovered:", len(unique_topics))
print("Topic IDs:", sorted(unique_topics))
