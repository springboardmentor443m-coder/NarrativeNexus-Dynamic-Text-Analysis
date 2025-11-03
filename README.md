# NarrativeNexus - Streamlit + FastAPI


### 1️⃣ Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate # or .venv\\Scripts\\activate (Windows)
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000


### 2️⃣ Frontend setup
streamlit run app.py