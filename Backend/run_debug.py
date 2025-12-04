import sys
print("Starting app import...")
try:
    from main import app
    print("App imported successfully")
    import uvicorn
    print("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
