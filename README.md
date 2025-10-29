To run this project 
---------------------
1.  python -m venv venv
2.  .\venv\Scripts\Activate.ps1
3.  pip install -r requirements.txt
4.  uvicorn backend.main:app --reload     (backend)
Now in new terminal run front end 
-----------------------------------------
streamlit run frontend/app.py

In .env and list_groq_models.py File Give Your Own GROQ_API_KEY