# Email/SMS Spam Classifier

A Streamlit application that classifies an email or SMS message as **spam** or
**ham** (legitimate). The model uses TF-IDF features and a Multinomial Naive
Bayes classifier trained on the included SMS spam dataset.

## Live demo

Try the deployed application: [SMS Spam Classifier](https://sms-spam-classifier31.streamlit.app/)

## Run locally

Use Python 3.12 or 3.14 with the pinned dependencies below. The saved model
and vectorizer were generated with scikit-learn 1.9.0; keep that version aligned
when retraining or deploying.

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

## Smoke tests

```bash
python3 -m unittest discover -s tests
```

These tests check fitted artifacts, matching scikit-learn versions, preprocessing,
startup, blank input, and legitimate/spam messages through the Streamlit interface.
An optional GitHub Actions configuration is in `ci/classifier-checks.yml.example`.
To enable automated checks on Python 3.12 and 3.14, copy it to
`.github/workflows/tests.yml` using credentials with GitHub workflow permissions.

## Streamlit Community Cloud

Deploy the `main` branch with `app.py` as the entrypoint. Keep `requirements.txt`
at the repository root and install all its dependencies, not just Streamlit.
The first prediction downloads NLTK's small English stopwords corpus if missing;
later predictions reuse it. This first request therefore needs internet access.

If the deployed page reports `ModuleNotFoundError` even after a dependency fix
was pushed, open **Manage app**, check the build log, and choose **Reboot app**
from its menu to force a fresh environment. Refreshing the browser alone does
not reinstall missing packages. See the [Streamlit reboot guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).

PyArrow is pinned to 24.0.0 to match the working cloud environment; the September
2026 cloud build log replaced 25.0.1 automatically because of a known crash.

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
