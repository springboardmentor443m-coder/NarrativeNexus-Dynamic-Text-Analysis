# ==============================================================
# 🧠 Generate sample text dataset from 20 Newsgroups
# ==============================================================

from sklearn.datasets import fetch_20newsgroups
import os
import random

# 1️⃣ Fetch the dataset (removes headers, footers, quotes for cleaner text)
print("Downloading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

texts = data.data
targets = data.target
target_names = data.target_names

print(f"✅ Loaded {len(texts)} documents across {len(target_names)} topics.")
print("Example categories:", target_names[:5])

# 2️⃣ Create an output folder
output_dir = "sample_newsgroup_docs"
os.makedirs(output_dir, exist_ok=True)

# 3️⃣ Randomly pick 10 documents to save
num_docs = 10
selected_indices = random.sample(range(len(texts)), num_docs)

for i, idx in enumerate(selected_indices):
    category = target_names[targets[idx]]
    filename = f"{output_dir}/doc_{i+1}_{category.replace('/', '_')}.txt"

    # Clean text (optional: basic cleanup for readability)
    text = texts[idx].replace('\n', ' ').replace('\r', ' ').strip()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved: {filename}")

print("\n🎉 Done! 10 sample text files created in the 'sample_newsgroup_docs/' folder.")
print("Each file represents a different topic. You can upload them into your Streamlit app.")
