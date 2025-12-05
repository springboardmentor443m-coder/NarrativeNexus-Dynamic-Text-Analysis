"""
Training script for LDA theme extraction model on 20 Newsgroups dataset
This script downloads the dataset, trains an LDA model, and saves it for later use.
"""
import os
import pickle
import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def clean_and_prepare_text(text):
    """Clean and prepare text for theme modeling"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses and URLs
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) 
             for token in tokens 
             if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def build_theme_model():
    """Build LDA model on 20 Newsgroups dataset"""
    print("=" * 60)
    print("Building Theme Extraction Model on 20 Newsgroups Dataset")
    print("=" * 60)
    
    # Create model storage directory if it doesn't exist
    os.makedirs('model_storage', exist_ok=True)
    
    # Step 1: Download 20 Newsgroups dataset
    print("\n[1/5] Downloading 20 Newsgroups dataset...")
    print("This may take a few minutes on first run...")
    
    try:
        newsgroups_train = fetch_20newsgroups(
            subset='train',
            shuffle=True,
            random_state=42,
            remove=('headers', 'footers', 'quotes')
        )
        print(f"✓ Downloaded {len(newsgroups_train.data)} training documents")
        print(f"✓ Dataset contains {len(newsgroups_train.target_names)} categories")
        print(f"  Categories: {', '.join(newsgroups_train.target_names)}")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False
    
    # Step 2: Preprocess the data
    print("\n[2/5] Preprocessing documents...")
    print("This may take a few minutes...")
    
    processed_documents = []
    for i, doc in enumerate(newsgroups_train.data):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(newsgroups_train.data)} documents...")
        processed_doc = clean_and_prepare_text(doc)
        if len(processed_doc.split()) > 10:  # Skip very short documents
            processed_documents.append(processed_doc)
    
    print(f"✓ Preprocessed {len(processed_documents)} documents")
    
    # Step 3: Create document-term matrix
    print("\n[3/5] Creating document-term matrix...")
    
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    doc_term_matrix = vectorizer.fit_transform(processed_documents)
    print(f"✓ Created matrix with shape: {doc_term_matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    # Step 4: Train LDA model
    print("\n[4/5] Training LDA model...")
    print("This may take several minutes...")
    
    # Number of themes = number of categories in 20 newsgroups
    num_themes = len(newsgroups_train.target_names)
    
    lda_model = LatentDirichletAllocation(
        n_components=num_themes,
        random_state=42,
        max_iter=20,
        learning_method='batch',
        n_jobs=-1
    )
    
    lda_model.fit(doc_term_matrix)
    print(f"✓ Trained LDA model with {num_themes} themes")
    
    # Step 5: Save the model and related data
    print("\n[5/5] Saving model and data...")
    
    model_data = {
        'lda_model': lda_model,
        'vectorizer': vectorizer,
        'topic_names': newsgroups_train.target_names,
        'num_topics': num_themes,
        'feature_names': vectorizer.get_feature_names_out()
    }
    
    model_path = os.path.join('model_storage', 'lda_model_20newsgroups.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to: {model_path}")
    
    # Display sample themes
    print("\n" + "=" * 60)
    print("Sample Themes (Top 10 words per theme):")
    print("=" * 60)
    
    feature_names = vectorizer.get_feature_names_out()
    for theme_idx, theme in enumerate(lda_model.components_):
        top_words_idx = theme.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"\nTheme {theme_idx + 1} ({newsgroups_train.target_names[theme_idx]}):")
        print(f"  {' '.join(top_words)}")
    
    print("\n" + "=" * 60)
    print("Model building completed successfully!")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print("You can now run the application with: python server.py")
    
    return True

if __name__ == '__main__':
    success = build_theme_model()
    if not success:
        print("\nModel building failed. Please check the error messages above.")
        exit(1)
