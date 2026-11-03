Step 1: cd ~/projects/phishing-detector

Step 2: Activate the Python Virtual Environment

       =source .venv/bin/activate


Step 3:python train.py



Step 4: Start the FastAPI Backend Server (Terminal Tab 1)

            =uvicorn app:app --port 8000






Step 5: Launch the Streamlit Web Interface (Terminal Tab 2)

              = cd ~/projects/phishing-detector 
                source .venv/bin/activate
                streamlit run ui.py


-----------------------------------------------------------------* * *---------------------------------------------------------------------



Step 6: Test and Analyze URLs in Your Browser


1= Open your web browser and go to http://localhost:8501.   


2= Paste any test URL into the input 


3= Click the analyze button to view real-time verdicts
