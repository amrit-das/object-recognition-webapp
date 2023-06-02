# Streamlit to run the app
import streamlit as st

# Python Packages
import os
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_icon=":raised_hand_with_fingers_splayed:",
    page_title="Object Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Object Recognition WebApp")

st.sidebar.header("Hyperparameter")

conf = float(st.sidebar.slider("Model Confidence", 0, 100, 60)) / 100

model_path = 