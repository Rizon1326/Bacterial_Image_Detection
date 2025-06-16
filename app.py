import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import cv2
import time

# Import custom utilities
from utils.model_utils import load_model, predict_image
from utils.visualization import display_results, plot_confidence_distribution
from utils.metrics import calculate_metrics, plot_precision_recall_curve, plot_f1_score

# Page configuration
st.set_page_config(
    page_title="🦠 YOLOv8 Bacterial Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 YOLOv8 Bacterial Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Unleash the power of AI to detect bacteria in images with cutting-edge precision!</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model settings
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            help="Lower values detect more objects but may include false positives"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="Higher values reduce overlapping detections"
        )
        
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.info("Upload an image to see detailed performance metrics!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image containing bacteria for detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
    
    with col2:
        if uploaded_file is not None:
            st.header("🔍 Detection Results")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Load model
                status_text.text("Loading model...")
                progress_bar.progress(20)
                model = load_model("models/bacterial_detection_final.pt")
                
                # Run prediction
                status_text.text("Running detection...")
                progress_bar.progress(60)
                results = predict_image(
                    model, 
                    temp_image_path, 
                    conf_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Display results
                status_text.text("Processing results...")
                progress_bar.progress(80)
                
                annotated_image, detection_data = display_results(
                    results, 
                    show_labels=show_labels,
                    show_confidence=show_confidence
                )
                
                progress_bar.progress(100)
                status_text.text("Detection completed!")
                
                # Show annotated image
                st.image(annotated_image, caption="Detection Results", use_column_width=True)
                
                # Clean up temp file
                os.unlink(temp_image_path)
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                return
    
    # Results section
    if uploaded_file is not None and 'detection_data' in locals():
        st.markdown("---")
        st.header("📊 Detection Analysis")
        
        # Detection summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🦠 Bacteria Detected",
                value=detection_data['total_detections'],
                delta=f"{detection_data['high_confidence_detections']} high confidence"
            )
        
        with col2:
            st.metric(
                label="🎯 Average Confidence",
                value=f"{detection_data['avg_confidence']:.2f}",
                help="Average confidence score of all detections"
            )
        
        with col3:
            st.metric(
                label="📏 Image Resolution",
                value=f"{detection_data['image_width']}x{detection_data['image_height']}"
            )
        
        with col4:
            st.metric(
                label="⚡ Processing Time",
                value=f"{detection_data['processing_time']:.2f}s"
            )
        
        # Detailed metrics and visualizations
        if detection_data['total_detections'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Confidence Distribution")
                confidence_fig = plot_confidence_distribution(detection_data['confidences'])
                st.plotly_chart(confidence_fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Performance Metrics")
                
                # Calculate and display metrics
                metrics = calculate_metrics(detection_data)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Model Performance</h4>
                    <p><strong>F1-Score:</strong> {metrics['f1_score']:.3f}</p>
                    <p><strong>Precision:</strong> {metrics['precision']:.3f}</p>
                    <p><strong>Recall:</strong> {metrics['recall']:.3f}</p>
                    <p><strong>mAP@0.5:</strong> {metrics['map_50']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced visualizations
            st.subheader("📊 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Precision-Recall Curve
                pr_fig = plot_precision_recall_curve(detection_data)
                st.plotly_chart(pr_fig, use_container_width=True)
            
            with col2:
                # F1-Score visualization
                f1_fig = plot_f1_score(detection_data)
                st.plotly_chart(f1_fig, use_container_width=True)
            
            # Detection details table
            st.subheader("🔍 Detection Details")
            if detection_data['detections']:
                import pandas as pd
                
                df = pd.DataFrame(detection_data['detections'])
                df.index += 1
                df.columns = ['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height']
                
                st.dataframe(
                    df.style.format({
                        'Confidence': '{:.3f}',
                        'X1': '{:.0f}',
                        'Y1': '{:.0f}',
                        'X2': '{:.0f}',
                        'Y2': '{:.0f}',
                        'Width': '{:.0f}',
                        'Height': '{:.0f}'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("No bacteria detected in the uploaded image. Try adjusting the confidence threshold.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🔬 Built with YOLOv8 | Perfect for researchers, scientists, and bacterial analysis enthusiasts</p>
            <p>🚀 Powered by Streamlit & Ultralytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
    
    main()