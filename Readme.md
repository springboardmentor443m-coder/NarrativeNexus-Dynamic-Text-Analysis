# 🚀 Quick Start: Connecting Frontend and Backend

## Option 1: Integrated Server (Easiest) ⭐

The easiest way to run both frontend and backend together:

### Step 1: Start the Server

*Windows:*
bash
python start_server.py


*Linux/Mac:*
bash
python3 start_server.py


Or use the quick start script:
- *Windows:* Double-click quick_start.bat
- *Linux/Mac:* Run chmod +x quick_start.sh && ./quick_start.sh

### Step 2: Use the Application

1. The browser will open automatically at http://localhost:8000/app
2. If not, navigate to http://localhost:8000/app manually
3. Enter text and click "Analyze Text"
4. View results in real-time!

## Option 2: Manual Setup

### Backend Only

bash
python main.py


Backend runs on: http://localhost:8000

### Access Frontend

1. *Via Integrated Server:* Visit http://localhost:8000/app
2. *Direct File:* Open frontend_example.html in your browser
3. *Separate Server:* Run python -m http.server 3000 and open http://localhost:3000/frontend_example.html

## Testing the Connection

### 1. Check Backend is Running

bash
curl http://localhost:8000/


### 2. Test API

bash
curl http://localhost:8000/api/stats


### 3. Open Frontend

Visit: http://localhost:8000/app

## URLs

- *Frontend App:* http://localhost:8000/app
- *API Root:* http://localhost:8000/
- *API Docs:* http://localhost:8000/docs
- *API Endpoints:* http://localhost:8000/api/*

## Troubleshooting

### Connection Issues

1. *Make sure backend is running:* Check http://localhost:8000/
2. *Check browser console:* Press F12 and look for errors
3. *Verify CORS:* Backend has CORS enabled by default
4. *Check port:* Default is 8000, make sure it's not in use

### Frontend Not Loading

1. Make sure frontend_example.html exists in the project directory
2. Check the file path in the browser
3. Verify the server is running

### API Errors

1. Check backend logs for errors
2. Verify the endpoint URL in the frontend
3. Check request format (Content-Type: application/json)

## Next Steps

1. *Customize:* Modify frontend_example.html to match your needs
2. *Integrate:* Use the API examples in FRONTEND_INTEGRATION.md
3. *Deploy:* Deploy to a cloud service for production

## Quick Reference

bash
# Start integrated server (recommended)
python start_server.py

# Start backend only
python main.py

# Test backend
curl http://localhost:8000/api/stats

# Open frontend
# Visit http://localhost:8000/app in browser


## File Structure


platform/
├── main.py                    # Backend API
├── start_server.py            # Integrated server (backend + frontend)
├── frontend_example.html      # Frontend application
├── CONNECTION_GUIDE.md        # Detailed connection guide
└── FRONTEND_INTEGRATION.md    # Frontend integration examples


## Support

For more detailed information, see:
- CONNECTION_GUIDE.md - Detailed connection instructions
- FRONTEND_INTEGRATION.md - Frontend integration examples
- README.md - Full project documentation
