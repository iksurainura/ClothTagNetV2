import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import time
import numpy as np
import queue
import av
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors  # Key import for overlay

# --- Page Config ---
st.set_page_config(page_title="YOLO Live Segmentation", layout="wide")
result_queue = queue.Queue()

@st.cache_resource
def load_models():
    return YOLO("yolo11n-seg.pt"), YOLO("yolo11m-cls.pt")

seg_model, cls_model = load_models()

def extract_and_warp(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    w, h = int(rect[1][0]), int(rect[1][1])
    if w == 0 or h == 0: return None, None
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, matrix, (w, h)), box

class SmartProcessor(VideoTransformerBase):
    def __init__(self):
        self.is_collecting = False
        self.collection_start = 0
        self.frame_buffer = []
        self.last_detection_time = 0
        self.COOLDOWN = 3.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        curr_time = time.time()
        
        # --- Real-time Segmentation Overlay ---
        # We perform a fast inference here just for the visual overlay
        overlay_results = seg_model(img, conf=0.5, verbose=False)[0]
        annotator = Annotator(img.copy())
        
        if overlay_results.masks is not None:
            for mask, box in zip(overlay_results.masks.xy, overlay_results.boxes):
                # Draw the mask and the bounding box on the live feed
                annotator.seg_bbox(mask=mask, mask_color=colors(int(box.cls), True), label=f"{seg_model.names[int(box.cls)]}")

        display_frame = annotator.result()

        # --- High-Quality Collection Logic ---
        if self.is_collecting:
            elapsed = curr_time - self.collection_start
            if elapsed <= 2.0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                self.frame_buffer.append({"score": score, "frame": img.copy()})
                cv2.putText(display_frame, f"PROCESSING QUALITY: {int((elapsed/2)*100)}%", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                self.is_collecting = False
                if self.frame_buffer:
                    best = max(self.frame_buffer, key=lambda x: x['score'])
                    results = seg_model(best['frame'], conf=0.7, verbose=False)[0]
                    if results.masks is not None:
                        mask_data = results.masks.data[0].cpu().numpy().astype(np.uint8)
                        extracted, _ = extract_and_warp(best['frame'], mask_data)
                        if extracted is not None:
                            cls_res = cls_model.predict(extracted, verbose=False)[0]
                            result_queue.put({
                                "img": extracted,
                                "label": cls_res.names[cls_res.probs.top1],
                                "conf": cls_res.probs.top1conf.item()
                            })
                self.frame_buffer = []
        
        elif curr_time - self.last_detection_time > self.COOLDOWN:
            if overlay_results.masks is not None:
                self.is_collecting = True
                self.collection_start = curr_time
                self.last_detection_time = curr_time

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

# --- UI Setup ---
st.title("Live Segmentation & Sharp-Capture Analysis")
c1, c2 = st.columns([1.5, 1])

with c1:
    st.markdown("### 🎥 Live Feed (with Segmentation)")
    ctx = webrtc_streamer(
        key="seg-stream",
        video_processor_factory=SmartProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with c2:
    st.markdown("### 📸 Best Sharp Frame")
    res_img = st.empty()
    res_text = st.empty()
    
    while ctx.state.playing:
        try:
            data = result_queue.get(timeout=1)
            res_img.image(cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB), use_container_width=True)
            res_text.success(f"**Classification:** {data['label']} ({data['conf']:.2%})")
        except queue.Empty:
            continue