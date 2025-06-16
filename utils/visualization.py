import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

def display_results(results, show_labels=True, show_confidence=True):
    """Process and display YOLO results"""
    
    if not results or len(results) == 0:
        return None, {'total_detections': 0}
    
    result = results[0]
    
    # Get annotated image
    annotated_img = result.plot(
        labels=show_labels,
        conf=show_confidence,
        line_width=2,
        font_size=12
    )
    
    # Convert BGR to RGB
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Extract detection data
    detection_data = extract_detection_data(result)
    
    return annotated_img_rgb, detection_data

def extract_detection_data(result):
    """Extract detailed detection information"""
    
    data = {
        'total_detections': 0,
        'high_confidence_detections': 0,
        'avg_confidence': 0,
        'confidences': [],
        'detections': [],
        'image_width': 0,
        'image_height': 0,
        'processing_time': getattr(result, 'processing_time', 0)
    }
    
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        
        # Image dimensions
        data['image_height'], data['image_width'] = result.orig_shape
        
        # Extract box data
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        
        data['total_detections'] = len(confidences)
        data['confidences'] = confidences.tolist()
        data['avg_confidence'] = float(np.mean(confidences))
        data['high_confidence_detections'] = int(np.sum(confidences > 0.5))
        
        # Detailed detection info
        for i in range(len(confidences)):
            x1, y1, x2, y2 = xyxy[i]
            detection = [
                f"Bacteria_{int(classes[i])}",
                float(confidences[i]),
                float(x1), float(y1), float(x2), float(y2),
                float(x2 - x1), float(y2 - y1)
            ]
            data['detections'].append(detection)
    
    return data

def plot_confidence_distribution(confidences):
    """Plot confidence score distribution"""
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#667eea']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        title_x=0.5
    )
    
    return fig