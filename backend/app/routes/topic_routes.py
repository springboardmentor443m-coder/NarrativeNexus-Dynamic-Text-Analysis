# ----------------------LDA & NMF TOPIC MODELING ROUTES-----------------------

# from fastapi import APIRouter, UploadFile
# from app.utils.topic_modelling import perform_topic_modeling
# from app.utils.sentiment_analysis import analyze_topic_sentiment
# from app.utils.preprocessing import clean_text , preprocess_pipeline
# router = APIRouter()

# @router.post("/analyze")
# async def analyze_text(file: UploadFile, method: str = "lda", num_topics: int = 5):
#     # Read uploaded file
#     raw_text = (await file.read()).decode("utf-8")
    
#     # Clean text
#     preprocessed_text = preprocess_pipeline(raw_text)
#     cleaned_docs = clean_text(preprocessed_text)

#     # Perform topic modeling
#     topics_result = perform_topic_modeling(cleaned_docs, num_topics=num_topics, method=method)
    
#     # Handle errors
#     if "error" in topics_result:
#         return {"error": topics_result["error"]}
    
#     # Perform sentiment analysis on the topics
#     topics_with_sentiment = analyze_topic_sentiment(topics_result["topics"])
    
#     return {
#         "method": topics_result["method"],
#         "num_topics": topics_result["num_topics"],
#         "topics": topics_with_sentiment
#     }


################# WORKING VERSION 1
# ----------------------TRANSFORMER-BASED TOPIC MODELING ROUTES-----------------------


# from fastapi import APIRouter, UploadFile
# from app.utils.preprocessing import preprocess_pipeline
# from app.utils.topic_modelling import perform_topic_modeling_transformers
# from app.utils.sentiment_analysis import analyze_topic_sentiment
# from app.utils.summarization import summarize_with_groq  # 👈 new Groq-based summarizer

# router = APIRouter()

# @router.post("/analyze")
# async def analyze_text(file: UploadFile, num_topics: int = 5):
#     # 1️⃣ Read and preprocess
#     raw_text = (await file.read()).decode("utf-8")
#     cleaned_text = preprocess_pipeline(raw_text)

#     # 2️⃣ Topic modeling
#     topics_result = perform_topic_modeling_transformers(cleaned_text, num_topics=num_topics)
#     if "error" in topics_result:
#         return {"error": topics_result["error"]}

#     # 3️⃣ Sentiment analysis
#     topics_with_sentiment = analyze_topic_sentiment(topics_result["topics"])

#     # 4️⃣ Summarization using Groq API
#     summaries = []
#     for topic in topics_with_sentiment:
#         text = topic.get("text", "")
#         if not text.strip():
#             summaries.append("")
#             continue

#         summary = summarize_with_groq(text, max_length=100, min_length=30)
#         summaries.append(summary)

#     # 5️⃣ Combine results
#     return {
#         "method": topics_result["method"],
#         "num_topics": topics_result["num_topics"],
#         "topics": topics_with_sentiment,
#         "summary": summaries
#     }


##########################version 2 with topic wise summary

from fastapi import APIRouter, UploadFile, HTTPException
from app.utils.preprocessing import preprocess_pipeline
from app.utils.topic_modelling import perform_topic_modeling_transformers
from app.utils.sentiment_analysis import analyze_topic_sentiment
from app.utils.summarization import summarize_with_groq

router = APIRouter()

@router.post("/analyze")
async def analyze_text(file: UploadFile, num_topics: int = 5):
    """
    Upload a text file and perform:
    1. Preprocessing (cleaning, normalization)
    2. Topic modeling using Transformer embeddings
    3. Sentiment analysis for each topic (via Groq API)
    4. Summarization of each topic
    """
    try:
        # Step 1: Read and preprocess text
        raw_text = (await file.read()).decode("utf-8")
        cleaned_text = preprocess_pipeline(raw_text)

        # Step 2: Topic modeling
        topics_result = perform_topic_modeling_transformers(cleaned_text, num_topics=num_topics)
        if "error" in topics_result:
            raise HTTPException(status_code=400, detail=topics_result["error"])

        # Step 3: Sentiment analysis
        topics_with_sentiment = analyze_topic_sentiment(topics_result["topics"])

        # Step 4: Summarization for each topic
        final_topics = []
        for topic in topics_with_sentiment:
            topic_text = topic.get("text", "")
            sentiment = topic.get("sentiment", {})

            # Generate summary for topic
            summary = summarize_with_groq(topic_text, max_length=80, min_length=30)

            # Append structured result
            final_topics.append({
                "topic_id": topic.get("topic_id"),
                "keywords": topic.get("keywords", []),
                "text": topic_text,
                "sentiment": sentiment,
                "summary": summary
            })

        # Step 5: Return final structured response
        return {
            "method": topics_result.get("method", "Transformers"),
            "num_topics": topics_result.get("num_topics", len(final_topics)),
            "topics": final_topics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
