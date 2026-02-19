# main.py
"""
Conveyor Belt Label Inspection System
Run with: streamlit run main.py
"""

import streamlit as st
import cv2
import torch
import numpy as np
from datetime import datetime
import threading
import time
from pathlib import Path

# Import our modules
from defect_detector import DefectDetectionOCR, SystemState
from video_stream import VideoStream
from config import CONFIG

# Page configuration
st.set_page_config(
    page_title="Label Inspection System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Added compact video styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
        display: inline-block;
        font-size: 0.9rem;
    }
    .status-idle { background-color: #28a745; color: white; }
    .status-capturing { background-color: #ffc107; color: black; }
    .status-processing { background-color: #dc3545; color: white; }
    .compact-video {
        max-height: 400px !important;
        width: auto !important;
        margin: 0 auto;
    }
    .video-container {
        display: flex;
        justify-content: center;
        background-color: #000;
        border-radius: 0.5rem;
        padding: 0.5rem;
        max-height: 420px;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-box {
        background-color: #e9ecef;
        border-radius: 0.3rem;
        padding: 0.4rem;
        text-align: center;
        font-size: 0.9rem;
    }
    .small-font {
        font-size: 0.85rem;
    }
    .sharpest-frame {
        max-height: 200px !important;
        width: auto !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'video_stream' not in st.session_state:
        st.session_state.video_stream = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'last_sharpest_frame' not in st.session_state:
        st.session_state.last_sharpest_frame = None
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = []

def load_models():
    """Load YOLO models"""
    try:
        from ultralytics import YOLO
        
        # Load segmentation model (for real-time detection)
        seg_model = YOLO(CONFIG['seg_model_path'])
        
        # Load OBB model (for oriented bounding box detection)
        obb_model = YOLO(CONFIG['obb_model_path'])
        
        return seg_model, obb_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

def add_log(message: str):
    """Add timestamped log message"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}"
    st.session_state.system_logs.append(log_entry)
    # Keep only last 100 logs
    if len(st.session_state.system_logs) > 100:
        st.session_state.system_logs = st.session_state.system_logs[-100:]

def draw_overlay(frame, mask, corners, status):
    """Draw segmentation overlay on frame"""
    vis = frame.copy()
    
    if mask is not None:
        # Create colored mask overlay
        colored_mask = np.zeros_like(vis)
        colored_mask[mask > 0] = [0, 255, 0]  # Green mask
        vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.3, 0)
        
        # Draw mask contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    if corners is not None:
        # Draw corner points
        corners = np.array(corners, dtype=np.int32)
        for i, point in enumerate(corners):
            cv2.circle(vis, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw connecting lines
        cv2.polylines(vis, [corners], True, (255, 0, 0), 2)
    
    # Add status text
    status_colors = {
        'IDLE': (0, 255, 0),
        'CAPTURING': (0, 255, 255),
        'PROCESSING': (0, 0, 255)
    }
    color = status_colors.get(status, (128, 128, 128))
    cv2.putText(vis, f"STATE: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return vis

def resize_frame(frame, max_height=400):
    """Resize frame to have maximum height while maintaining aspect ratio"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = max_height
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def main():
    st.markdown('<div class="main-header">🏭 Label Inspection System</div>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar controls - more compact
    with st.sidebar:
        st.header("⚙️ Controls", divider="gray")
        
        # Model paths
        seg_path = st.text_input("Seg Model", CONFIG['seg_model_path'], label_visibility="collapsed")
        st.caption("Segmentation Model Path")
        obb_path = st.text_input("OBB Model", CONFIG['obb_model_path'], label_visibility="collapsed")
        st.caption("OBB Model Path")
        source = st.text_input("Video Source", CONFIG['video_source'], label_visibility="collapsed")
        st.caption("0=Webcam, or path/URL")
        
        # Parameters in expander to save space
        with st.expander("🔧 Parameters", expanded=False):
            capture_duration = st.slider("Capture (s)", 0.5, 5.0, 2.0, 0.1)
            min_area = st.slider("Min Area", 0.001, 0.1, 0.015, 0.001)
            max_area = st.slider("Max Area", 0.5, 1.0, 0.92, 0.01)
            conf_threshold = st.slider("Conf", 0.1, 0.9, 0.57, 0.01)
            
            CONFIG['capture_duration'] = capture_duration
            CONFIG['min_area_ratio'] = min_area
            CONFIG['max_area_ratio'] = max_area
            CONFIG['seg_conf_threshold'] = conf_threshold
        
        # Update config
        CONFIG['seg_model_path'] = seg_path
        CONFIG['obb_model_path'] = obb_path
        CONFIG['video_source'] = source
        
        # Control buttons - compact row
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ Start", use_container_width=True):
                if not st.session_state.is_running:
                    seg_model, obb_model = load_models()
                    if seg_model and obb_model:
                        st.session_state.detector = DefectDetectionOCR(
                            seg_model, obb_model, capture_duration
                        )
                        st.session_state.video_stream = VideoStream(source)
                        st.session_state.is_running = True
                        st.session_state.video_stream.start()
                        add_log("System started")
                        st.rerun()
        
        with btn_col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.is_running = False
                if st.session_state.video_stream:
                    st.session_state.video_stream.stop()
                add_log("System stopped")
                st.rerun()
        
        # System status
        st.divider()
        status_col1, status_col2 = st.columns([1, 2])
        with status_col1:
            st.markdown("**Status:**")
        with status_col2:
            if st.session_state.detector:
                state = st.session_state.detector.state
                state_class = f"status-{state.name.lower()}"
                st.markdown(f'<span class="status-badge {state_class}">{state.name}</span>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge" style="background-color: #6c757d;">OFFLINE</span>', 
                           unsafe_allow_html=True)
        
        # Logs - compact
        st.divider()
        st.markdown("**📋 Logs**")
        log_container = st.container(height=150)
        with log_container:
            for log in reversed(st.session_state.system_logs[-15:]):
                st.text(log, help=None)
    
    # Main content area
    if not st.session_state.is_running or not st.session_state.detector:
        st.info("👈 Configure settings and click Start to begin inspection")
        return
    
    # Create compact layout - video on left (smaller), details on right
    col_video, col_details = st.columns([1, 1])
    
    with col_video:
        # Compact video feed
        st.markdown("**📹 Live Feed**")
        video_placeholder = st.empty()
        
        # Compact metrics in a row
        metric_cols = st.columns(4)
        with metric_cols[0]:
            total_inspected = st.metric("Count", 0, label_visibility="collapsed")
            st.caption("Inspected", help="Total labels inspected")
        with metric_cols[1]:
            fps_metric = st.metric("FPS", 0, label_visibility="collapsed")
            st.caption("FPS")
        with metric_cols[2]:
            buffer_size = st.metric("Buf", 0, label_visibility="collapsed")
            st.caption("Buffer")
        with metric_cols[3]:
            sharpness_metric = st.metric("Sharp", 0.0, label_visibility="collapsed")
            st.caption("Sharpness")
    
    with col_details:
        # Sharpest frame - compact
        st.markdown("**🎯 Best Frame**")
        sharpest_placeholder = st.empty()
        
        # OBB - compact
        st.markdown("**📐 OBB**")
        obb_placeholder = st.empty()
        
        # OCR Result - compact
        st.markdown("**📝 OCR**")
        ocr_placeholder = st.empty()
    
    # Results table - full width below
    st.divider()
    st.markdown("**📊 Recent Results**")
    results_table = st.empty()
    
    # Main processing loop
    detector = st.session_state.detector
    video_stream = st.session_state.video_stream
    
    frame_time = time.time()
    
    while st.session_state.is_running:
        # Get frame from stream
        frame = video_stream.read()
        if frame is None:
            time.sleep(0.001)
            continue
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - frame_time)
        frame_time = current_time
        
        # Process frame through detector
        status_data = detector.update(frame)
        
        # Create visualization
        vis_frame = draw_overlay(
            frame, 
            status_data.get('seg_mask'),
            status_data.get('seg_corners'),
            status_data['status']
        )
        
        # Resize for compact display
        vis_frame_resized = resize_frame(vis_frame, max_height=380)
        
        # Convert BGR to RGB for Streamlit
        vis_frame_rgb = cv2.cvtColor(vis_frame_resized, cv2.COLOR_BGR2RGB)
        
        # Update live feed with use_column_width to control size
        video_placeholder.image(
            vis_frame_rgb, 
            channels="RGB", 
            use_container_width=True
        )
        
        # Update metrics
        total_inspected.metric("Count", len(detector.get_all_results()))
        fps_metric.metric("FPS", f"{fps:.1f}")
        buffer_size.metric("Buf", status_data.get('buffer_count', 0))
        sharpness_metric.metric("Sharp", f"{status_data.get('sharpness', 0):.0f}")
        
        # Check for new results
        latest_result = detector.get_latest_result()
        if latest_result:
            result_id = latest_result['id']
            # Check if we haven't shown this result yet
            if not st.session_state.results_history or \
               st.session_state.results_history[-1]['id'] != result_id:
                
                st.session_state.results_history.append(latest_result)
                add_log(f"Result #{result_id}: '{latest_result['ocr_result'][:20]}...' ")
                
                # Update sharpest frame display - compact size
                warped = latest_result['warped_frame']
                if warped is not None and warped.size > 0:
                    # Resize for compact display
                    warped_resized = resize_frame(warped, max_height=180)
                    warped_rgb = cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB)
                    sharpest_placeholder.image(
                        warped_rgb, 
                        channels="RGB", 
                        use_container_width=True
                    )
                    st.session_state.last_sharpest_frame = warped_rgb
                
                # Draw OBB on warped frame - compact
                obb_frame = warped.copy()
                if latest_result['obb_corners'] is not None:
                    obb_pts = np.array(latest_result['obb_corners'], dtype=np.int32)
                    cv2.polylines(obb_frame, [obb_pts], True, (0, 255, 0), 2)
                    for pt in obb_pts:
                        cv2.circle(obb_frame, tuple(pt), 3, (0, 0, 255), -1)
                
                # Resize OBB frame
                obb_resized = resize_frame(obb_frame, max_height=180)
                obb_rgb = cv2.cvtColor(obb_resized, cv2.COLOR_BGR2RGB)
                obb_placeholder.image(
                    obb_rgb, 
                    channels="RGB", 
                    use_container_width=True
                )
                
                # Show OCR result - compact
                ocr_text = latest_result['ocr_result'] or "No text"
                confidence_color = "green" if len(ocr_text) > 3 else "orange"
                ocr_placeholder.markdown(f"""
                <div style="padding: 0.6rem; background-color: #f8f9fa; border-radius: 0.3rem; border-left: 3px solid {confidence_color};">
                    <span style="font-size: 1.1rem; font-weight: bold; color: {confidence_color};">{ocr_text}</span>
                    <br>
                    <span style="font-size: 0.75rem; color: #666;">Sharp: {latest_result['sharpness']:.1f} | ID: {result_id}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Update results table - compact
        if st.session_state.results_history:
            import pandas as pd
            df_data = []
            for r in reversed(st.session_state.results_history[-8:]):  # Last 8 only
                time_str = r['timestamp'].split('T')[1][:8] if 'T' in r['timestamp'] else r['timestamp'][-8:]
                text_str = r['ocr_result'][:25] + '..' if len(r['ocr_result']) > 25 else r['ocr_result']
                df_data.append({
                    'ID': r['id'],
                    'Time': time_str,
                    'Text': text_str,
                    'Sharp': f"{r['sharpness']:.0f}",
                    '✓': '✓' if r['ocr_result'] else '✗'
                })
            results_table.dataframe(
                pd.DataFrame(df_data), 
                use_container_width=True, 
                hide_index=True,
                height=200
            )
        
        # Small delay to prevent UI freezing
        time.sleep(0.03)
        
        # Check if we should stop
        if not st.session_state.is_running:
            break

if __name__ == "__main__":
    main()