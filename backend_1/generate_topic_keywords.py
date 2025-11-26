import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

with open("backend/models/topic_model/documents.pkl", "rb") as f:
    documents = pickle.load(f)

labels = np.load("backend/models/topic_model/topic_labels.npy")

print("Documents loaded:", len(documents))
print("Unique topics:", len(set(labels)))


# --------------------------------------------------------------------
# 1. Prepare dataframe
# --------------------------------------------------------------------
df = pd.DataFrame({"text": documents, "topic": labels})


# --------------------------------------------------------------------
# 2. Group documents by topic (join into one big doc per topic)
# --------------------------------------------------------------------
topic_docs = df.groupby("topic")["text"].apply(lambda texts: " ".join(texts))


# --------------------------------------------------------------------
# 3. Compute c-TF-IDF
# --------------------------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
count_matrix = vectorizer.fit_transform(topic_docs)

tfidf = TfidfTransformer()
ctfidf_matrix = tfidf.fit_transform(count_matrix)

feature_names = vectorizer.get_feature_names_out()


# --------------------------------------------------------------------
# 4. Extract top N keywords per topic
# --------------------------------------------------------------------
def get_top_keywords(row, n=12):
    indices = row.toarray()[0].argsort()[-n:][::-1]
    return [feature_names[i] for i in indices]


topic_keywords = {}
for topic_id, row in zip(topic_docs.index, ctfidf_matrix):
    topic_keywords[int(topic_id)] = get_top_keywords(row, n=12)

# --------------------------------------------------------------------
# 5. Save keywords
# --------------------------------------------------------------------
with open("backend/models/topic_model/topic_keywords.pkl", "wb") as f:
    pickle.dump(topic_keywords, f)

print("\n🎉 Keyword extraction complete!")
print("Saved to backend/models/topic_model/topic_keywords.pkl\n")

# Show sample
print("Example:")
for tid, kws in list(topic_keywords.items())[:5]:
    print(f"Topic {tid}: {kws}")
