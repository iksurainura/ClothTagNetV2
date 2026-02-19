# utils.py
"""
Utility functions for image processing and visualization
"""

import cv2
import numpy as np
from datetime import datetime


def draw_detections(frame, mask, corners, text="", color=(0, 255, 0)):
    """Draw segmentation mask and corners on frame"""
    output = frame.copy()
    
    if mask is not None:
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = color
        output = cv2.addWeighted(output, 1.0, overlay, 0.3, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
    
    if corners is not None:
        corners = np.array(corners, dtype=np.int32)
        # Draw points
        for i, pt in enumerate(corners):
            cv2.circle(output, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(output, str(i), tuple(pt), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw polygon
        cv2.polylines(output, [corners], True, (255, 0, 0), 2)
    
    if text:
        cv2.putText(output, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return output


def create_obb_visualization(image, corners, text=""):
    """Draw OBB detection results"""
    vis = image.copy()
    
    if corners is not None:
        pts = np.array(corners, dtype=np.int32)
        # Draw box
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        # Draw corners
        for pt in pts:
            cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
    
    if text:
        # Add text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (10, 10), (10 + tw, 10 + th + 10), (0, 0, 0), -1)
        cv2.putText(vis, text, (10, 10 + th), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return vis


def save_inspection_result(result, output_dir):
    """Save inspection result images to disk"""
    from pathlib import Path
    import json
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base_path = Path(output_dir) / f"inspection_{result['id']:04d}_{timestamp}"
    
    # Save original frame
    if result.get('frame') is not None:
        cv2.imwrite(str(base_path) + "_original.jpg", result['frame'])
    
    # Save warped frame
    if result.get('warped_frame') is not None:
        cv2.imwrite(str(base_path) + "_warped.jpg", result['warped_frame'])
    
    # Save metadata
    metadata = {
        'id': result['id'],
        'timestamp': result['timestamp'],
        'ocr_result': result['ocr_result'],
        'sharpness': result['sharpness'],
        'seg_corners': result['seg_corners'],
        'obb_corners': result['obb_corners']
    }
    
    with open(str(base_path) + "_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return base_path


def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    # Convert to polygon format if needed
    poly1 = np.array(box1).reshape(-1, 2)
    poly2 = np.array(box2).reshape(-1, 2)
    
    # Calculate intersection using contour area
    ret, intersection = cv2.intersectConvexConvex(poly1, poly2)
    if not ret:
        return 0.0
    
    intersection_area = cv2.contourArea(intersection)
    
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0