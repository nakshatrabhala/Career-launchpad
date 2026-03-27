# Career Launchpad Founder Analytics App

A flat-file Streamlit project for descriptive, diagnostic, predictive, and prescriptive analytics on the UAE student career app survey dataset.

## Files
- `app.py` — main Streamlit application
- `requirements.txt` — Python dependencies for Streamlit deployment
- `uae_student_career_app_synthetic_data.xlsx` — bundled training dataset
- `new_customers_template.csv` — template for future scoring uploads
- `README.md` — setup notes

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload these files to the root of a GitHub repository.
2. Make sure `app.py` is the entry file.
3. Deploy the repo on Streamlit Community Cloud.

## App capabilities
- Descriptive analytics
- Diagnostic analytics
- K-Means and latent-class style clustering
- Classification with:
  - accuracy
  - precision
  - recall
  - F1-score
  - ROC curve
  - feature importance
- Regression for monthly expenditure
- Association rule mining with:
  - support
  - confidence
  - lift
- Future customer upload and scoring
