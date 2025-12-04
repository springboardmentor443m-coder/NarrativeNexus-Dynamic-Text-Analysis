import sys
try:
    from main import app
    print("MAIN LOADED OK")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
