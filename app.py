import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# TRAINING DATA
# =========================

texts = [
    "Whoever commits murder shall be punished",
    "The accused committed robbery and theft",
    "Kidnapping and criminal assault case",
    "Fraud and cheating under IPC section",
    "Punishment for attempted murder",

    "Property dispute between two families",
    "Land ownership disagreement case",
    "Divorce and family settlement issue",
    "Civil rights violation petition",
    "Compensation claim for property damage",

    "Company violated shareholder agreement",
    "Corporate tax fraud investigation",
    "Business contract breach case",
    "Illegal activities in private company",
    "Financial audit of corporation"
]

labels = [
    "Criminal",
    "Criminal",
    "Criminal",
    "Criminal",
    "Criminal",

    "Civil",
    "Civil",
    "Civil",
    "Civil",
    "Civil",

    "Corporate",
    "Corporate",
    "Corporate",
    "Corporate",
    "Corporate"
]

# =========================
# MODEL TRAINING
# =========================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

model = MultinomialNB()

model.fit(X, labels)

# =========================
# UI DESIGN
# =========================

st.set_page_config(
    page_title="Legal Document Classifier",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legal Document Classification")

st.write(
    "Enter legal document text and predict its category."
)

user_input = st.text_area(
    "Enter Legal Text",
    height=200
)

if st.button("Predict Category"):

    if user_input.strip() != "":

        transformed_text = vectorizer.transform([user_input])

        prediction = model.predict(transformed_text)

        st.success(f"Predicted Category: {prediction[0]}")

    else:
        st.warning("Please enter some legal text.")
