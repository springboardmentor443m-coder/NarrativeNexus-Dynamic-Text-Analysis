from backend_1.topic_model import topic_model

# Get topic info as a dataframe
info = topic_model.get_topic_info()

# Exclude the "-1" noise topic
valid_topics = info[info["Topic"] != -1]

print("Number of topics discovered:", len(valid_topics))
print(valid_topics.head())
