# CROP-GUARD
A free, farmer-friendly crop health app that analyzes leaf photos, identifies likely diseases, and provides simple, source-backed care guidance in local languages.
# Crop Guard 🌱

A farmer-friendly crop health prototype that analyzes a tomato leaf photo and shows a possible disease match with simple care guidance.

## Current features

- Upload a tomato leaf photo
- View the model’s top crop and disease matches
- See basic, source-linked early blight care guidance when the result meets the app’s checks
- Record leaf symptoms, watering, and when the problem started

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py