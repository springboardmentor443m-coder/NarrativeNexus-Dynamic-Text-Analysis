#!/usr/bin/env python
import sys
import traceback

print("[1] Testing imports...")
try:
    print("  - Importing FastAPI...")
    from fastapi import FastAPI
    print("    [OK]")
    
    print("  - Importing staticfiles...")
    from fastapi.staticfiles import StaticFiles
    print("    [OK]")
    
    print("  - Importing Jinja2Templates...")
    from fastapi.templating import Jinja2Templates
    print("    [OK]")
    
    print("  - Importing modules...")
    from preprocessing import clean_text
    from sentiment_analysis import create_dummy_model
    from topic_modeling import create_dummy_lda_model
    print("    [OK]")
    
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[2] Testing staticfiles mount...")
try:
    from pathlib import Path
    static_dir = Path("static")
    print(f"  - Static dir exists: {static_dir.exists()}")
    print(f"  - Static path: {static_dir.absolute()}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n[3] Testing templates...")
try:
    from pathlib import Path
    templates_dir = Path("templates")
    print(f"  - Templates dir exists: {templates_dir.exists()}")
    print(f"  - Templates path: {templates_dir.absolute()}")
    if templates_dir.exists():
        print(f"  - Files: {list(templates_dir.glob('*'))}")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n[4] Creating FastAPI app...")
try:
    app = FastAPI(title="Test")
    print("  [OK]")
except Exception as e:
    print(f"[ERROR] {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[5] Mounting static files...")
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("  [OK]")
except Exception as e:
    print(f"[ERROR] {e}")
    traceback.print_exc()

print("\n[6] Creating templates...")
try:
    templates = Jinja2Templates(directory="templates")
    print("  [OK]")
except Exception as e:
    print(f"[ERROR] {e}")
    traceback.print_exc()

print("\n[7] Importing main.py...")
try:
    import main
    print("  [OK]")
except Exception as e:
    print(f"[ERROR] Failed to import main: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All checks passed!")
print("\nTo start server, run:")
print("  python -m uvicorn main:app --host 0.0.0.0 --port 8000")
