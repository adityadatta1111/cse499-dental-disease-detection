from ultralytics import YOLO
import streamlit as st
import settings
import cv2

def load_model(model_path):
    model = YOLO(model_path)
    return model