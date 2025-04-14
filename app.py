 
# pip install streamlit اعتقد عندي 
# streamlit run app.py 

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load your trained model and tokenizer
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Labels 
labels = ['Negative', 'Neutral', 'Positive']
icons = ['😠', '😐', '😃']
colors = ['#FF4B4B', '#FFA500', '#4CAF50']

# Streamlit page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# Title and description
st.title("Sentiment Analysis App (Distil Model)")
st.write("Enter your product review:")

# Text input
user_input = st.text_area("Write your review......")

# Predict button
if st.button("Analyze Sentiment!"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()

        # Display result
        st.markdown(f"## {icons[pred]} <span style='color:{colors[pred]}'>{labels[pred]}</span>", unsafe_allow_html=True)

        # Confidence scores
        st.subheader("📊 Confidence Scores:")
        for i, label in enumerate(labels):
            st.markdown(f"**{label}**: {probs[0][i].item():.2f}")

    else:
        st.warning("Please enter a review first.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using distil model and Streamlit.")