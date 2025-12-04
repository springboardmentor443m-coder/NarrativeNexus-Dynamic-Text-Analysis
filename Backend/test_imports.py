try:
    import fastapi
    print("OK: FastAPI installed")
except ImportError as e:
    print(f"ERROR: FastAPI missing: {e}")

try:
    import uvicorn
    print("OK: Uvicorn installed")
except ImportError as e:
    print(f"ERROR: Uvicorn missing: {e}")

try:
    import sklearn
    print("OK: scikit-learn installed")
except ImportError as e:
    print(f"ERROR: scikit-learn missing: {e}")

try:
    import transformers
    print("OK: transformers installed")
except ImportError as e:
    print(f"ERROR: transformers missing: {e}")

try:
    import torch
    print("OK: torch installed")
except ImportError as e:
    print(f"ERROR: torch missing: {e}")
