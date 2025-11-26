import os
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def safe_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass