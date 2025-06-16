import time
from ultralytics import YOLO
import streamlit as st

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run prediction on image"""
    start_time = time.time()
    
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