import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import precision_recall_curve, f1_score

def calculate_metrics(detection_data):
    """Calculate performance metrics"""
    
    # Simulated ground truth for demonstration
    # In real scenario, you'd have actual ground truth data
    
    confidences = np.array(detection_data['confidences'])
    
    if len(confidences) == 0:
        return {
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'map_50': 0.0
        }
    
    # Simulated metrics (replace with actual ground truth comparison)
    # These are example calculations
    precision = np.mean(confidences > 0.3)  # Simple threshold-based precision
    recall = len(confidences) / max(1, len(confidences) + 2)  # Simulated recall
    f1 = 2 * (precision * recall) / max(0.001, precision + recall)
    map_50 = np.mean(confidences) * 0.8  # Simulated mAP
    
    return {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'map_50': float(map_50)
    }

def plot_precision_recall_curve(detection_data):
    """Plot Precision-Recall curve"""
    
    confidences = np.array(detection_data['confidences'])
    
    if len(confidences) == 0:
        # Empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No detections available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Precision-Recall Curve")
        return fig
    
    # Generate sample PR curve data
    thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        pred_positive = confidences > threshold
        if np.sum(pred_positive) == 0:
            precision = 1.0
            recall = 0.0
        else:
            # Simulated precision and recall
            precision = np.mean(confidences[pred_positive] > 0.3)
            recall = np.sum(pred_positive) / len(confidences)
        
        precisions.append(precision)
        recalls.append(recall)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recalls,
        y=precisions,
        mode='lines+markers',
        name='PR Curve',
        line=dict(color='#667eea', width=3),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
        title_x=0.5
    )
    
    return fig

def plot_f1_score(detection_data):
    """Plot F1-Score vs Threshold"""
    
    confidences = np.array(detection_data['confidences'])
    
    if len(confidences) == 0:
        # Empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No detections available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="F1-Score vs Threshold")
        return fig
    
    thresholds = np.linspace(0.1, 0.9, 30)
    f1_scores = []
    
    for threshold in thresholds:
        pred_positive = confidences > threshold
        if np.sum(pred_positive) == 0:
            f1 = 0.0
        else:
            # Simulated F1 calculation
            precision = np.mean(confidences[pred_positive] > 0.3)
            recall = np.sum(pred_positive) / len(confidences)
            f1 = 2 * (precision * recall) / max(0.001, precision + recall)
        
        f1_scores.append(f1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_scores,
        mode='lines+markers',
        name='F1-Score',
        line=dict(color='#764ba2', width=3),
        marker=dict(size=6)
    ))
    
    # Highlight maximum F1
    max_f1_idx = np.argmax(f1_scores)
    fig.add_trace(go.Scatter(
        x=[thresholds[max_f1_idx]],
        y=[f1_scores[max_f1_idx]],
        mode='markers',
        name=f'Max F1: {f1_scores[max_f1_idx]:.3f}',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title="F1-Score vs Confidence Threshold",
        xaxis_title="Confidence Threshold",
        yaxis_title="F1-Score",
        height=400,
        title_x=0.5
    )
    
    return fig