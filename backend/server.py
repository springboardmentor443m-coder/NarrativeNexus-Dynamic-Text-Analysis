"""
Server startup script that serves both the backend API and frontend
This script starts the server and opens the browser automatically
"""
import uvicorn
import webbrowser
import time
import threading
from pathlib import Path

# Import the main app (which already has the /app route configured)
from main import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    # Try /app first, fallback to root
    webbrowser.open('http://localhost:8000/')

if _name_ == "_main_":
    print("=" * 60)
    print("Starting Dynamic Text Analysis Platform")
    print("=" * 60)
    print("\n✓ Backend API: http://localhost:8000")
    print("✓ API Documentation: http://localhost:8000/docs")
    print("✓ Frontend App: http://localhost:8000/app")
    print("\n🚀 Browser will open automatically in 2 seconds...")
    print("Press Ctrl+C to stop the server\n")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload on code changes
        log_level="info"
    )
