from pathlib import Path
import pickle

import streamlit as st

from text_processing import transform_text


PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    with (PROJECT_DIR / "vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)
    with (PROJECT_DIR / "model.pkl").open("rb") as file:
        model = pickle.load(file)
    return vectorizer, model


tfidf, model = load_artifacts()

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message", placeholder="Type or paste a message here")

if st.button("Classify", type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("Spam")
        else:
            st.success("Ham")
