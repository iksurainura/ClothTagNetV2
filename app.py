"""
Real-time YOLOv8 Segmentation App
- Left: Live webcam feed
- Right: Segmentation prediction overlay
- Bottom: Confirmed prediction table (label stable for 2s → logged with timestamp)

Install deps:
    pip install streamlit ultralytics opencv-python-headless Pillow pandas
Run:
    streamlit run seg_app.py
"""

import time
import datetime
import threading
import queue

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Seg · Live",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — dark GitHub-flavoured theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem; }
    header[data-testid="stHeader"] { display: none; }
    #MainMenu, footer { visibility: hidden; }

    .stApp { background: #0d1117; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }

    .logo-bar {
        font-family: 'Syne', sans-serif;
        font-size: 28px; font-weight: 800; color: #58a6ff;
        padding: 14px 0 6px; border-bottom: 1px solid #21262d;
        margin-bottom: 16px; letter-spacing: -0.5px;
    }
    .logo-bar span { color: #3fb950; }

    .status-pill {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px;
        border: 1px solid; font-family: 'JetBrains Mono', monospace;
    }
    .pill-idle    { background:#161b22; color:#8b949e; border-color:#30363d; }
    .pill-detect  { background:#2a2200; color:#d29922; border-color:#d29922; }
    .pill-done    { background:#0d2a1a; color:#3fb950; border-color:#3fb950; }

    .section-header {
        font-size: 11px; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1.5px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 5px; margin: 14px 0 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    .confirmed-badge {
        display: inline-block; padding: 4px 14px; border-radius: 6px;
        font-size: 13px; font-weight: 700;
        background: #0d2a1a; color: #3fb950; border: 1px solid #3fb950;
        font-family: 'JetBrains Mono', monospace;
    }
    .none-badge {
        display: inline-block; padding: 4px 14px; border-radius: 6px;
        font-size: 13px; font-weight: 700;
        background: #161b22; color: #8b949e; border: 1px solid #30363d;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 14px 18px;
    }
    .metric-label {
        font-size: 10px; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-val {
        font-size: 26px; font-weight: 700; color: #58a6ff;
        margin-top: 4px; font-family: 'Syne', sans-serif;
    }

    /* Table styling */
    .stDataFrame { border: 1px solid #21262d !important; border-radius: 8px; overflow: hidden; }
    .stDataFrame thead th {
        background: #161b22 !important; color: #8b949e !important;
        font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #21262d !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stDataFrame tbody td {
        background: #0d1117 !important; color: #e0e0e0 !important;
        font-size: 13px !important; border-bottom: 1px solid #161b22 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stDataFrame tbody tr:hover td { background: #161b22 !important; }

    /* Feed labels */
    .feed-label {
        font-size: 11px; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;
    }

    /* Stability bar container */
    .stab-wrap {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 6px; height: 8px; margin-top: 8px; overflow: hidden;
    }
    .stab-fill {
        height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, #d29922, #3fb950);
        transition: width 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "log" not in st.session_state:
    st.session_state.log = []          # list of {Timestamp, Prediction, Confidence}
if "model" not in st.session_state:
    st.session_state.model = None

# ─────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("trained_yolo11m-seg.pt")   # downloads automatically on first run

# ─────────────────────────────────────────────
# Segmentation helper
# ─────────────────────────────────────────────
PALETTE = [
    (255, 99,  71), (30, 144, 255), (50, 205,  50), (255, 215,   0),
    (186,  85, 211),(255, 165,   0), (0,  206, 209), (220,  20,  60),
    (127, 255,   0), (255,  20, 147),
]

def run_segmentation(frame, model):
    """Returns (annotated_frame_rgb, top_label, top_conf)."""
    results = model(frame, verbose=False)[0]

    annotated = frame.copy()
    top_label = None
    top_conf  = 0.0
    counts    = {}

    if results.masks is not None:
        masks   = results.masks.data.cpu().numpy()   # (N, H, W)
        boxes   = results.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        names   = model.names

        overlay = annotated.copy()
        for i, (mask, cls, conf) in enumerate(zip(masks, classes, confs)):
            color = PALETTE[cls % len(PALETTE)]
            label = names[cls]

            # Resize mask to frame size
            m = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            m = (m > 0.5).astype(np.uint8)

            overlay[m == 1] = (
                overlay[m == 1] * 0.45 + np.array(color) * 0.55
            ).astype(np.uint8)

            # Contour
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            # Label above bbox
            x1, y1 = int(boxes.xyxy[i][0]), int(boxes.xyxy[i][1])
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, txt, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)

            # Track dominant label by confidence sum
            counts[label] = counts.get(label, 0) + conf

        # Blend overlay
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        if counts:
            top_label = max(counts, key=counts.get)
            top_conf  = confs[classes == list(model.names.values()).index(top_label)
                              if top_label in list(model.names.values()) else 0].max() \
                        if len(confs) else 0.0
            # simpler: just pick label with highest summed conf
            top_conf = round(counts[top_label] / sum(1 for c in classes if model.names[c] == top_label), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, top_label, top_conf

# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
st.markdown('<div class="logo-bar">⬡ Seg<span>Live</span></div>', unsafe_allow_html=True)

# Top bar: controls + status
ctrl_col, stat_col, cnt_col = st.columns([2, 3, 2])

with ctrl_col:
    start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = st.button("⏹ Stop",  use_container_width=True)

with stat_col:
    status_ph = st.empty()

with cnt_col:
    count_ph = st.empty()

st.markdown('<div class="section-header">Live Feeds</div>', unsafe_allow_html=True)

# Two feed columns
feed_left, feed_right = st.columns(2)

with feed_left:
    st.markdown('<div class="feed-label">📷 Raw Camera</div>', unsafe_allow_html=True)
    raw_ph = st.empty()

with feed_right:
    st.markdown('<div class="feed-label">🧠 Segmentation Output</div>', unsafe_allow_html=True)
    seg_ph = st.empty()

# Stability bar
stab_ph = st.empty()

# Confirmed label
st.markdown('<div class="section-header">Confirmed Label</div>', unsafe_allow_html=True)
conf_label_ph = st.empty()

# Log table
st.markdown('<div class="section-header">Prediction Log</div>', unsafe_allow_html=True)
table_ph = st.empty()

# ─────────────────────────────────────────────
# Button logic
# ─────────────────────────────────────────────
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ─────────────────────────────────────────────
# Placeholder frames
# ─────────────────────────────────────────────
def blank_frame(text="Waiting…"):
    img = np.full((360, 480, 3), 13, dtype=np.uint8)
    cv2.putText(img, text, (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2)
    return img

if not st.session_state.running:
    raw_ph.image(blank_frame("Press ▶ Start"), channels="BGR", use_container_width=True)
    seg_ph.image(blank_frame("No feed yet"),   channels="BGR", use_container_width=True)
    status_ph.markdown('<span class="status-pill pill-idle">⬤ IDLE</span>', unsafe_allow_html=True)
    count_ph.markdown(
        '<div class="metric-card"><div class="metric-label">Total Logged</div>'
        f'<div class="metric-val">{len(st.session_state.log)}</div></div>',
        unsafe_allow_html=True
    )
    conf_label_ph.markdown('<span class="none-badge">— none confirmed —</span>', unsafe_allow_html=True)
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        table_ph.dataframe(df, use_container_width=True, hide_index=True)
    st.stop()

# ─────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────
model = load_model()
cap   = cv2.VideoCapture(1)

if not cap.isOpened():
    st.error("❌ Cannot open webcam. Make sure a camera is connected.")
    st.stop()

status_ph.markdown('<span class="status-pill pill-detect">⬤ RUNNING</span>', unsafe_allow_html=True)

# Stability tracking
stability_start = None   # time when current streak began
stable_label    = None   # current streak label
STABILITY_SEC   = 2.0

confirmed_label = None

try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        # Raw feed
        raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_ph.image(raw_rgb, use_container_width=True)

        # Segmentation
        seg_rgb, top_label, top_conf = run_segmentation(frame, model)
        seg_ph.image(seg_rgb, use_container_width=True)

        now = time.time()

        # Stability logic
        if top_label is not None:
            if top_label == stable_label:
                elapsed = now - stability_start
            else:
                stable_label    = top_label
                stability_start = now
                elapsed         = 0.0

            pct = min(elapsed / STABILITY_SEC, 1.0)

            stab_ph.markdown(
                f'<div style="margin:4px 0 2px;font-size:11px;color:#8b949e;'
                f'font-family:JetBrains Mono,monospace;">Stability: '
                f'<b style="color:#d29922">{top_label}</b> '
                f'— {elapsed:.1f}s / {STABILITY_SEC}s</div>'
                f'<div class="stab-wrap"><div class="stab-fill" style="width:{pct*100:.0f}%"></div></div>',
                unsafe_allow_html=True
            )

            # Confirm after 2 seconds
            if elapsed >= STABILITY_SEC:
                if top_label != confirmed_label:
                    confirmed_label = top_label
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    st.session_state.log.append({
                        "Timestamp":  ts,
                        "Prediction": top_label,
                        "Confidence": f"{top_conf:.2f}",
                    })
                    stability_start = now   # reset streak so we don't spam
        else:
            stab_ph.markdown(
                '<div style="font-size:11px;color:#8b949e;margin:4px 0;">No detections…</div>',
                unsafe_allow_html=True
            )
            stable_label    = None
            stability_start = None

        # Update confirmed label display
        if confirmed_label:
            conf_label_ph.markdown(
                f'<span class="confirmed-badge">✔ {confirmed_label}</span>',
                unsafe_allow_html=True
            )

        # Update count card
        count_ph.markdown(
            '<div class="metric-card"><div class="metric-label">Total Logged</div>'
            f'<div class="metric-val">{len(st.session_state.log)}</div></div>',
            unsafe_allow_html=True
        )

        # Update table
        if st.session_state.log:
            df = pd.DataFrame(st.session_state.log[::-1])   # newest first
            table_ph.dataframe(df, use_container_width=True, hide_index=True)

finally:
    cap.release()
    status_ph.markdown('<span class="status-pill pill-idle">⬤ STOPPED</span>', unsafe_allow_html=True)