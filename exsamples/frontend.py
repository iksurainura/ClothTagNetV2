import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(page_title="YOLO Smart Capture", layout="wide")
st.title("🎯 YOLO Segmentation & Classification")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    # Loading nano/medium models as per your script
    seg = YOLO("yolo11n-seg.pt")
    cls = YOLO("yolo11m-cls.pt")
    return seg, cls

seg_model, cls_model = load_models()

# --- Helper Logic ---
def get_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_and_warp(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    width, height = int(rect[1][0]), int(rect[1][1])
    if width == 0 or height == 0: return None, None
    
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped, box

# --- Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.is_collecting = False
        self.collection_start = 0
        self.frame_buffer = []
        self.last_detection_time = 0
        self.display_result_until = 0
        
        # Stored results for UI overlay
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_box = None
        
        # Settings
        self.BUFFER_TIME = 2.0
        self.CONF_THRESH = 0.7
        self.COOLDOWN = 2.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        display_frame = img.copy()

        if self.is_collecting:
            elapsed = current_time - self.collection_start
            if elapsed <= self.BUFFER_TIME:
                score = get_sharpness(img)
                self.frame_buffer.append({"score": score, "frame": img.copy()})
                
                # Visual Feedback
                progress = int((elapsed / self.BUFFER_TIME) * 100)
                cv2.putText(display_frame, f"Capturing: {progress}%", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Processing Stage
                self.is_collecting = False
                if self.frame_buffer:
                    best = max(self.frame_buffer, key=lambda x: x['score'])
                    results = seg_model(best['frame'], conf=self.CONF_THRESH, verbose=False)[0]
                    
                    if results.masks is not None:
                        mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
                        extracted, box = extract_and_warp(best['frame'], mask)
                        
                        if extracted is not None:
                            cls_res = cls_model.predict(extracted, verbose=False)[0]
                            self.current_prediction = cls_res.names[cls_res.probs.top1]
                            self.current_confidence = cls_res.probs.top1conf.item()
                            self.current_box = box
                            self.display_result_until = current_time + 4.0
                self.frame_buffer = []

        elif current_time - self.last_detection_time > self.COOLDOWN:
            # Monitor Mode
            results = seg_model(img, conf=self.CONF_THRESH, verbose=False)[0]
            if results.masks is not None:
                self.is_collecting = True
                self.collection_start = current_time
                self.last_detection_time = current_time

        # Draw Persistent UI
        if current_time < self.display_result_until and self.current_box is not None:
            cv2.drawContours(display_frame, [self.current_box], 0, (0, 255, 0), 3)
            label = f"{self.current_prediction} ({self.current_confidence:.1%})"
            cv2.putText(display_frame, label, (self.current_box[0][0], self.current_box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame.from_ndarray(display_frame, format="bgr24")

# --- UI Sidebar & Layout ---
with st.sidebar:
    st.header("Controls")
    st.write("The system monitors for objects. Once detected, it captures 2 seconds of video and selects the sharpest frame for classification.")
    st.info("Models used: YOLO11n-seg & YOLO11m-cls")

webrtc_streamer(
    key="yolo-filter",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)