import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import joblib, json, os
from utils import preprocess_text

def train(input_csv, text_col, n_topics=10, max_features=2000, model_out_dir='backend/models'):
    os.makedirs(model_out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    texts = df[text_col].fillna('').astype(str).tolist()
    cleaned = [preprocess_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=max_features)
    X = vectorizer.fit_transform(cleaned)
    model = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=200)
    W = model.fit_transform(X)
    H = model.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for t_idx, topic_vec in enumerate(H):
        topn = topic_vec.argsort()[-15:][::-1]
        keywords = [feature_names[i] for i in topn]
        topics[str(t_idx)] = {"keywords": keywords}
    joblib.dump(vectorizer, f"{model_out_dir}/vectorizer.joblib")
    joblib.dump(model, f"{model_out_dir}/topic_model.joblib")
    with open(f"{model_out_dir}/topics.json","w",encoding='utf-8') as f:
        json.dump(topics, f, indent=2)
    print('Saved vectorizer, model and topics to', model_out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--n-topics', type=int, default=10)
    parser.add_argument('--max-features', type=int, default=2000)
    args = parser.parse_args()
    train(args.input_csv, args.text_column, n_topics=args.n_topics, max_features=args.max_features)
