from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel


def build_lda_model(tokenized_texts, num_topics=5):
    # Build bigrams and trigrams
    bigram = Phrases(tokenized_texts, min_count=5, threshold=50)
    trigram = Phrases(bigram[tokenized_texts], threshold=50)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    tokenized_texts = [trigram_mod[bigram_mod[doc]] for doc in tokenized_texts]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # Build LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        eta='auto'
    )

    # Evaluate coherence
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()

    topics = []
    for idx, topic in lda_model.print_topics(-1):
        keywords = [w.split('*')[1].replace('\"', '').strip() for w in topic.split('+')]
        topics.append({'topic_id': idx, 'keywords': keywords})

    return {
        'topics': topics,
        'coherence': coherence_score
    }
