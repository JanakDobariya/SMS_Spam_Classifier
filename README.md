# Email/SMS Spam Classifier

A Streamlit application that classifies an email or SMS message as **spam** or
**ham** (legitimate). The model uses TF-IDF features and a Multinomial Naive
Bayes classifier trained on the included SMS spam dataset.

## Run locally

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Streamlit will print the local address where the app is available, normally
<http://localhost:8501>.

## Retrain the model

```bash
python3 train_model.py
```

This evaluates the classifier on a held-out test set, retrains it with all
available messages, and regenerates `model.pkl` and `vectorizer.pkl`.

## Project structure

```text
.
├── Data/spam.csv               # Training data
├── app.py                      # Streamlit interface
├── text_processing.py          # Shared text preprocessing
├── train_model.py              # Reproducible training script
├── sms_spam_detection.ipynb    # Exploratory analysis and experiments
├── model.pkl                   # Fitted classifier
└── vectorizer.pkl              # Fitted TF-IDF vectorizer
```

The current held-out evaluation produces approximately 97.7% accuracy and
100% precision for the spam class. Results may vary if the data, preprocessing,
or dependency versions are changed.
