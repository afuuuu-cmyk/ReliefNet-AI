import streamlit as st
import pickle
import os

# Correct path to model
MODEL_PATH = os.path.join("model", "relief_model.pkl")

# Use new Streamlit caching method
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Make sure 'relief_model.pkl' exists inside the 'model/' folder.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# UI
st.title("🆘 ReliefNet Help Categorizer")
st.write("Enter a help request message, and we'll predict its category.")

text_input = st.text_area("Enter request (e.g., 'I need urgent blood help')")

if st.button("Predict Category"):
    if text_input.strip() == "":
        st.warning("Please enter a request.")
    else:
        pred = model.predict([text_input])[0]
        st.success(f"✅ Predicted Category: **{pred}**")
