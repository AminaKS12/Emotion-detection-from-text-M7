import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Emotion to Emoji mapping
emotion_emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🧠 Emotion Detection from Text")
st.write("Type a sentence below to predict the emotion it conveys.")

# Input box
user_input = st.text_area("Enter your sentence here:")

if st.button("Detect Emotion"):
    if user_input.strip():
        # Vectorize and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)

        # Get confidence
        confidence = np.max(proba) * 100

        # Show result
        st.markdown(f"### Predicted Emotion: **{prediction.capitalize()}** {emotion_emoji.get(prediction, '')}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter a sentence.")
