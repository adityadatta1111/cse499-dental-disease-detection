from pathlib import Path
import PIL
import streamlit as st

st.set_page_config(
    page_title="Dental Image Segmentation and Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dental Image Detection and Prediction Using YOLOv8 and SAM")

st.sidebar.header("DL Model Configuration")