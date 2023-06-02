# Streamlit to run the app
import streamlit as st

# Python Packages
from pathlib import Path

# Custom Packages
from utils import load, from_webcam, from_image
import config

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

# Model Definition
model = load(config.MODEL_PATH / config.DET_MODEL_NAME)

src = st.sidebar.radio("Recognize from:", ("Webcam", "Image", "Default"), index=2)

if src == "Default":
    pass
# Capture Video From WebCam

if src == "Webcam":
    video = from_webcam(conf, model)
elif src == "Image":
    image = from_image(conf, model)
else:
    pass
