# backend_1/list_topics.py

import pickle
from backend_1.topic_model import topic_model

# Topic info
info = topic_model.get_topic_info()
valid = info[info["Topic"] != -1]

print(f"Total topics: {len(valid)}\n")

for _, row in valid.iterrows():
    tid = int(row["Topic"])
    rep = topic_model.get_topic(tid)
    keywords = [w for w, _ in rep] if rep else []

    print(f"Topic {tid}:")
    print(" Keywords:", ", ".join(keywords[:10]))
    print("-" * 60)
