import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import threading
from queue import Queue
from typing import Optional
from DefectInspection import DefectInspection, SystemConfig, InspectionResult

# ==================== MODEL CONFIGURATION ====================
# Set your model path here
MODEL_PATH = "models/YOLO11m-seg.pt"  # Change this to your model path
# =============================================================

# Page configuration
st.set_page_config(
    page_title="Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inspection_system' not in st.session_state:
    st.session_state.inspection_system = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'thread' not in st.session_state:
    st.session_state.thread = None
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = Queue(maxsize=2)
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = Queue(maxsize=10)

class StreamlitDefectInspection(DefectInspection):
    """Extended DefectInspection class for Streamlit integration"""
    
    def __init__(self, config: SystemConfig = None, model_path: str = "", 
                 frame_queue: Queue = None, result_queue: Queue = None):
        super().__init__(config, model_path)
        self.start = False
        self.frame_queue = frame_queue
        self.result_queue = result_queue
    
    def run_continuous(self):
        """Modified run method that continuously sends frames and results to queues"""
        if not hasattr(self, 'obb_model'):
            raise RuntimeError("Model not loaded. Call load_models() first.")
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not started. Call start_camera() first.")
        
        self.start = True
        print("Starting continuous inspection loop...")

        while self.start:
            self.frame_buffer = []
            cycle_start_time = time.time()
            
            # Capture frames for buffer_time seconds
            while self.start and (time.time() - cycle_start_time) < self.config.buffer_time:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    self.start = False
                    break
                
                # Send current frame to queue for display
                if self.frame_queue:
                    try:
                        # Clear old frames and add new one
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                break
                        self.frame_queue.put_nowait(frame.copy())
                    except Exception as e:
                        pass
                
                # Calculate sharpness and store
                sharpness = self.Sharpest_frame(frame)
                self.frame_buffer.append({
                    'frame': frame.copy(),
                    'sharpness': sharpness,
                })
            
            if not self.start or len(self.frame_buffer) == 0:
                break
            
            # Get sharpest frame
            best = max(self.frame_buffer, key=lambda x: x['sharpness'])
            sharpest_frame = best['frame'].copy()
            
            print(f"Processing sharpest frame (sharpness: {best['sharpness']:.2f})")
            
            # Run model prediction on sharpest frame
            if hasattr(self, 'obb_model'):
                try:
                    results = self.obb_model.predict(
                        sharpest_frame,
                        conf=self.config.seg_conf,
                        verbose=False
                    )
                    
                    # Draw bounding boxes on a copy
                    annotated_frame = sharpest_frame.copy()
                    detection_made = False
                    detected_label = "no_detection"
                    detected_conf = 0.0
                    
                    # Check if we have OBB or regular boxes
                    has_detections = False
                    boxes_data = None
                    
                    if len(results) > 0:
                        # Try OBB first (Oriented Bounding Boxes)
                        if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
                            boxes_data = results[0].obb
                            has_detections = True
                            print(f"OBB detections found: {len(boxes_data)}")
                        # Try regular boxes
                        elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes_data = results[0].boxes
                            has_detections = True
                            print(f"Regular box detections found: {len(boxes_data)}")
                    
                    if has_detections and boxes_data is not None:
                        best_idx = boxes_data.conf.argmax().item()
                        
                        # Draw all bounding boxes
                        for i in range(len(boxes_data)):
                            conf = float(boxes_data.conf[i])
                            cls = int(boxes_data.cls[i])
                            label = results[0].names[cls]
                            
                            # Use different color for best detection
                            color = (0, 255, 0) if i == best_idx else (255, 165, 0)
                            thickness = 3 if i == best_idx else 2
                            
                            # Handle OBB (rotated boxes)
                            if hasattr(boxes_data, 'xyxyxyxy'):
                                # OBB - draw rotated rectangle
                                points = boxes_data.xyxyxyxy[i].cpu().numpy().astype(int)
                                cv2.polylines(annotated_frame, [points], True, color, thickness)
                                
                                # Put label at first point
                                label_text = f"{label}: {conf:.2f}"
                                cv2.putText(annotated_frame, label_text,
                                          (points[0][0], points[0][1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                          color, 2)
                            else:
                                # Regular boxes
                                xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
                                cv2.rectangle(annotated_frame, 
                                            (xyxy[0], xyxy[1]), 
                                            (xyxy[2], xyxy[3]), 
                                            color, thickness)
                                
                                label_text = f"{label}: {conf:.2f}"
                                cv2.putText(annotated_frame, label_text,
                                          (xyxy[0], xyxy[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                          color, 2)
                        
                        # Get best detection info
                        detected_label = results[0].names[int(boxes_data.cls[best_idx])]
                        detected_conf = float(boxes_data.conf[best_idx])
                        detection_made = True
                        
                        print(f"Detection: {detected_label} ({detected_conf:.2f})")
                    else:
                        print("No detections found")
                    
                    # Create result
                    result = InspectionResult(
                        label=detected_label,
                        confidence=detected_conf,
                        extracted_image=annotated_frame
                    )
                    
                    # Send result to queue
                    if self.result_queue:
                        result_data = {
                            'result': result,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'detection_made': detection_made
                        }
                        try:
                            self.result_queue.put_nowait(result_data)
                            print(f"Result added to queue. Queue size: {self.result_queue.qsize()}")
                        except Exception as e:
                            print(f"Failed to add result to queue: {e}")
                            
                except Exception as e:
                    print(f"Prediction error: {e}")
                    import traceback
                    traceback.print_exc()

def run_inspection_thread(inspection_system):
    """Thread function to run inspection"""
    try:
        inspection_system.run_continuous()
    except Exception as e:
        print(f"Error in inspection thread: {e}")
        import traceback
        traceback.print_exc()

# Main UI
st.markdown('<p class="main-header">🔍 Real-Time Defect Inspection System</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    camera_source = st.number_input("Camera Source", min_value=0, max_value=10, value=0, 
                                   help="Camera index (0 for default)")
    buffer_time = st.slider("Buffer Time (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                           help="Time to collect frames before selecting sharpest")
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                                    help="Minimum confidence for detection")
    
    st.divider()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("▶️ Start", type="primary", disabled=st.session_state.is_running)
    
    with col2:
        stop_button = st.button("⏹️ Stop", type="secondary", disabled=not st.session_state.is_running)
    
    st.divider()
    
    # Stats
    st.subheader("📊 Statistics")
    st.metric("Detection Count", len(st.session_state.detection_results))
    
    if st.session_state.detection_results:
        detections_only = [r for r in st.session_state.detection_results if r['detection_made']]
        st.metric("Defects Found", len(detections_only))
        
        if detections_only:
            avg_conf = sum(r['result'].confidence for r in detections_only) / len(detections_only)
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
    
    st.divider()
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        st.session_state.detection_results = []
        st.rerun()

# Handle Start button
if start_button and not st.session_state.is_running:
    try:
        config = SystemConfig(buffer_time=buffer_time, seg_conf=confidence_threshold)
        inspection = StreamlitDefectInspection(
            config=config,
            model_path=MODEL_PATH,
            frame_queue=st.session_state.frame_queue,
            result_queue=st.session_state.result_queue
        )
        
        with st.spinner("Loading model..."):
            inspection.load_models()
        
        with st.spinner("Starting camera..."):
            inspection.start_camera(source=camera_source)
        
        st.session_state.inspection_system = inspection
        st.session_state.is_running = True
        
        # Start inspection thread
        thread = threading.Thread(target=run_inspection_thread, args=(inspection,), daemon=True)
        thread.start()
        st.session_state.thread = thread
        
        st.success("✅ System started successfully!")
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error starting system: {e}")

# Handle Stop button
if stop_button and st.session_state.is_running:
    if st.session_state.inspection_system:
        st.session_state.inspection_system.stop()
        st.session_state.is_running = False
        st.success("✅ System stopped!")
        time.sleep(0.5)
        st.rerun()

# Main content area
if st.session_state.is_running:
    # Create two columns for live feed and results
    col_feed, col_results = st.columns([1, 1])
    
    with col_feed:
        st.subheader("📹 Live Camera Feed")
        frame_placeholder = st.empty()
    
    with col_results:
        st.subheader("🎯 Detection Results")
        results_container = st.container()
    
    # Get latest frame
    try:
        if not st.session_state.frame_queue.empty():
            frame = st.session_state.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    except Exception as e:
        pass
    
    # Get latest results
    try:
        while not st.session_state.result_queue.empty():
            result_data = st.session_state.result_queue.get_nowait()
            st.session_state.detection_results.insert(0, result_data)
            print(f"Result retrieved! Total results: {len(st.session_state.detection_results)}")
            
            # Keep only the latest result
            if len(st.session_state.detection_results) > 1:
                st.session_state.detection_results = st.session_state.detection_results[:1]
    except Exception as e:
        pass
    
    # Display results
    with results_container:
        if st.session_state.detection_results:
            for idx, result_data in enumerate(st.session_state.detection_results):
                result = result_data['result']
                timestamp = result_data['timestamp']
                detection_made = result_data['detection_made']
                
                # Status indicator
                if detection_made:
                    status_color = "🔴" if result.confidence > 0.8 else "🟡"
                    st.markdown(f"### {status_color} Defect Detected")
                else:
                    st.markdown(f"### ⚪ No Detection")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display image
                    img_rgb = cv2.cvtColor(result.extracted_image, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True, caption=f"Captured at {timestamp}")
                
                with col2:
                    st.markdown(f"**Time:** {timestamp}")
                    st.markdown(f"**Label:** {result.label}")
                    
                    if detection_made:
                        st.markdown(f"**Confidence:** {result.confidence:.2%}")
                        st.progress(result.confidence)
                    else:
                        st.markdown(f"**Confidence:** N/A")
                
                st.divider()
        else:
            st.info("⏳ Waiting for detections... System is processing frames.")
    
    # Auto-refresh to update display
    time.sleep(0.5)
    st.rerun()
    
else:
    # Show instructions when not running
    st.info("👈 Configure settings in the sidebar and click **Start** to begin inspection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📋 Instructions
        1. Set model path in the code (MODEL_PATH variable)
        2. Select camera source (usually 0)
        3. Adjust buffer time and confidence threshold
        4. Click **Start** to begin real-time inspection
        5. View live feed and detection results side-by-side
        6. Click **Stop** when finished
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Features
        - Real-time camera feed display
        - Automatic sharpest frame selection
        - Visual bounding box annotations
        - Confidence scores for each detection
        - Latest detection result
        - Live statistics tracking
        """)