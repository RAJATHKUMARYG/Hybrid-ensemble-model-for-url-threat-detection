# Hybrid Ensemble Model for URL Threat Detection

This project is a Flask web app that analyzes URLs to predict whether they're malicious (Malware) or benign, shows explanation of top features (SHAP), and includes a chatbot for guidance.

What's included
- `app.py` — Flask app and routes (login, register, menu, analyze endpoints, chatbot).
- `templates/` — HTML templates (updated UI and chat widget).
- `static/style.css` — Improved backgrounds, chat widget and UI styling.
- `train_model.py` — Script to (re)train the ensemble model and save `ensemble_model.pkl` and `label_encoder.pkl`.
- `dataset.csv` — Example dataset (used for training/evaluation).

Quick setup
1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Retrain the model with your dataset:

```powershell
python train_model.py
```

This will produce `ensemble_model.pkl` and `label_encoder.pkl` used by the app.

3. (Optional) Enable ChatGPT-powered answers (broader QA):
- Install `openai` (already in requirements) and set the `OPENAI_API_KEY` environment variable to your OpenAI API key.

4. Run the app:

```powershell
python app.py
```

Security notes and limitations
- The `forgot-password` flow is a placeholder and does not send emails.
- The chatbot will use OpenAI if `OPENAI_API_KEY` is set; otherwise it uses a deterministic FAQ/fuzzy fallback.
- The explanation uses SHAP for the random-forest estimator; ensure `shap` is installed.

If you want me to run the app here and check behavior, confirm and I'll run a local sanity check and report any errors. If you want better chatbot answers without using OpenAI, I can add local retrieval from a knowledge base or connect to a local LLM if you have one available.
