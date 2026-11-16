# Crop Guard 🌱

A farmer-friendly crop health prototype for checking tomato leaf photos and showing simple care guidance.

## Features

- Upload a tomato leaf photo for a possible disease match
- English, Hindi, and Marathi interface
- Farmer observation questions about leaf signs, watering, and when the issue started
- Source-linked early blight care guidance when the result passes the app's checks

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py