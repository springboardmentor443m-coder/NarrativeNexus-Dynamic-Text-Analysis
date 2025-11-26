import pickle
import numpy as np

# Load the saved documents you used for training
with open("backend/models/topic_model/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Load the topic labels assigned during training
labels = np.load("backend/models/topic_model/topic_labels.npy")

topic_id = 70   # The topic you got as output

# Get all documents that belong to topic 70
topic_docs = [doc for doc, label in zip(documents, labels) if label == topic_id]

print(f"Number of documents in Topic {topic_id}: {len(topic_docs)}")
print("-" * 80)

# Print first 5 sample documents
for i, doc in enumerate(topic_docs[:5]):
    print(f"\n--- Document {i+1} -----------------------------------------\n")
    print(doc[:1200])  # print first 1200 chars for readability
