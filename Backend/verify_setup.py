#!/usr/bin/env python
"""Verify that the project is properly set up."""

import sys

try:
    print("Verifying project setup...\n")
    
    print("1. Checking FastAPI...")
    from fastapi import FastAPI
    print("   [OK] FastAPI available")
    
    print("2. Checking Jinja2Templates...")
    from fastapi.templating import Jinja2Templates
    print("   [OK] Jinja2Templates available")
    
    print("3. Checking preprocessing module...")
    from preprocessing import clean_text, preprocess_for_lda
    print("   [OK] preprocessing module loaded")
    
    print("4. Checking sentiment_analysis module...")
    from sentiment_analysis import create_dummy_model, predict_labels
    print("   [OK] sentiment_analysis module loaded")
    
    print("5. Checking topic_modeling module...")
    from topic_modeling import create_dummy_lda_model, infer_topics_for_text
    print("   [OK] topic_modeling module loaded")
    
    print("6. Checking main.py...")
    import main
    print("   [OK] main.py loads successfully")
    
    print("\n[SUCCESS] All components are properly configured!")
    print("\nTo start the server, run:")
    print("  python -m uvicorn main:app --reload --port 8000")
    
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {e}")
    sys.exit(1)
