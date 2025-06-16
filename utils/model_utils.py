import time
from ultralytics import YOLO
import streamlit as st
import torch

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path, task='detect')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run prediction on image"""
    if model is None:
        st.error("Model not loaded properly. Cannot perform prediction.")
        return None
        
    start_time = time.time()
    
    try:
        results = model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        # Add processing time to results
        if results:
            results[0].processing_time = processing_time
        
        return results
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None