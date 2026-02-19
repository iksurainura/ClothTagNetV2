"""
DefectVision — Streamlit Frontend for DefectDetectionOCR
=========================================================
Run:  streamlit run app.py

Pipeline:
  Camera frame → SEG model → Gate policies (bounds + area + stability)
  → 2-second frame buffer → sharpest frame → OBB model + OCR → display result
"""

import time
import threading
import queue
import base64
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="DefectVision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── remove default top padding so nothing hides under header ── */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem; }
    header[data-testid="stHeader"] { display: none; }          /* hide top bar */
    #MainMenu, footer { visibility: hidden; }

    /* ── global ── */
    .stApp { background: #0d1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }

    /* ── logo bar (used in HTML) ── */
    .logo-bar {
        font-size: 26px; font-weight: 700; color: #58a6ff;
        padding: 14px 0 4px; border-bottom: 1px solid #21262d;
        margin-bottom: 16px;
    }

    /* ── status pills (used in HTML) ── */
    .status-pill {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px;
        border: 1px solid;
    }
    .pill-idle { background: #161b22; color: #8b949e; border-color: #30363d; }
    .pill-detect { background: #2a2200; color: #d29922; border-color: #d29922; }
    .pill-capture { background: #2d1b1b; color: #ff6b35; border-color: #ff6b35; }
    .pill-process { background: #0d1f33; color: #58a6ff; border-color: #58a6ff; }
    .pill-done { background: #0d2a1a; color: #3fb950; border-color: #3fb950; }

    /* ── section header (used in HTML) ── */
    .section-header {
        font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 5px; margin: 14px 0 10px;
    }

    /* ── metric card (used in HTML) ── */
    .metric-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px 18px; text-align: center;
    }
    .metric-label {
        font-size: 11px; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-val { 
        font-size: 28px; font-weight: 700; 
        margin-top: 4px; 
    }

    /* ── pipeline dots (used in HTML) ── */
    .pipe-row {
        display: flex; align-items: center; margin-bottom: 8px;
    }
    .pipe-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #30363d; margin-right: 12px;
        flex-shrink: 0;
    }
    .pipe-dot-active { background: #d29922; box-shadow: 0 0 8px #d29922; }
    .pipe-dot-capture { background: #ff6b35; box-shadow: 0 0 8px #ff6b35; }
    .pipe-dot-process { background: #58a6ff; box-shadow: 0 0 8px #58a6ff; }

    /* ── result card (used in HTML) ── */
    .result-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px; margin-bottom: 14px;
    }
    .result-ocr {
        background: #0d1117; border-left: 3px solid #58a6ff;
        border-radius: 0 6px 6px 0; padding: 10px 12px;
        font-family: monospace; font-size: 13px; color: #e0e0e0;
        word-break: break-all; min-height: 40px;
        margin: 8px 0;
    }

    /* ── gate log box (defined in CSS but not used in HTML, keeping for compatibility) ── */
    .gate-log {
        background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 12px; font-family: monospace; font-size: 11px;
        color: #8b949e; height: 220px; overflow-y: auto;
    }

    /* ── badge classes (defined in CSS but not used in HTML, keeping for compatibility) ── */
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px;
    }
    .badge-green  { background:#0d2a1a; color:#3fb950; border:1px solid #3fb950; }
    .badge-yellow { background:#2a2200; color:#d29922; border:1px solid #d29922; }
    .badge-blue   { background:#0d1f33; color:#58a6ff; border:1px solid #58a6ff; }
    .badge-red    { background:#2a0d0d; color:#f85149; border:1px solid #f85149; }

    /* ── pipeline step labels (defined in CSS but not used in HTML, keeping for compatibility) ── */
    .pipe-step {
        display: inline-block; background: #21262d;
        border-radius: 4px; padding: 2px 8px;
        font-size: 11px; color: #8b949e; margin-right: 4px;
    }
    .pipe-arrow { color: #3fb950; font-weight: 700; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  TUNABLE CONSTANTS  (mirrors backend)
# ──────────────────────────────────────────────────────────────────────────────
STABLE_FRAMES_NEEDED = 5
STABILITY_TOLERANCE  = 15
MIN_AREA_RATIO       = 0.02
MAX_AREA_RATIO       = 0.90
SEG_CONF_THRESHOLD   = 0.55
CAPTURE_DURATION     = 2      # seconds
CAMERA_INDEX         = 0      # change if needed

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":         False,
        "stage":           "idle",          # idle | detecting | capturing | processing | done
        "stable_count":    0,
        "area_ratio":      0.0,
        "in_bounds":       False,
        "sharpness":       0.0,
        "capture_progress":0.0,
        "results":         [],
        "frame_rgb":       None,            # latest annotated frame for display
        "warped_rgb":      None,            # warped tag image
        "latest_result":   None,
        "stop_event":      None,
        "thread":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ──────────────────────────────────────────────────────────────────────────────
#  DRAW SEG OVERLAY ON FRAME
# ──────────────────────────────────────────────────────────────────────────────
STATE_COLORS = {
    "idle":       (125, 133, 144),
    "detecting":  (245, 158, 11),
    "capturing":  (255, 107, 53),
    "processing": (0,   180, 216),
    "done":       (0,   229, 160),
}

def draw_overlay(frame: np.ndarray, corners: np.ndarray,
                 stage: str, stable_count: int) -> np.ndarray:
    """Draw seg mask corners, bracket lines and status label on frame."""
    vis   = frame.copy()
    color = STATE_COLORS.get(stage, (125, 133, 144))

    if corners is None or len(corners) != 4:
        return vis

    pts = corners.astype(np.int32)

    # Semi-transparent fill
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)

    # Border  (dashed approximated by drawing)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

    # Corner brackets
    bsz = 16
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i - 1) % 4]
        da = _unit(b - a, bsz)
        dc = _unit(c - a, bsz)
        cv2.line(vis, tuple(a), (int(a[0]+da[0]), int(a[1]+da[1])), color, 2)
        cv2.line(vis, tuple(a), (int(a[0]+dc[0]), int(a[1]+dc[1])), color, 2)

    # Label badge
    if stage == "detecting":
        label = f"STABLE {stable_count}/{STABLE_FRAMES_NEEDED}"
    elif stage == "capturing":
        label = "BUFFERING 2s"
    elif stage == "processing":
        label = "PROCESSING"
    else:
        label = "DETECTED"

    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].min()) - 14
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(vis, (cx - tw//2 - 8, cy - th - 6),
                       (cx + tw//2 + 8, cy + 4), color, -1)
    cv2.putText(vis, label, (cx - tw//2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 12, 15), 1, cv2.LINE_AA)

    return vis


def _unit(v, length):
    n = np.linalg.norm(v)
    if n == 0: return np.array([0, 0])
    return (v / n * length).astype(int)

# ──────────────────────────────────────────────────────────────────────────────
#  BACKGROUND WORKER THREAD
# ──────────────────────────────────────────────────────────────────────────────
def _encode_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR → RGB for Streamlit display."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def detection_worker(detector, stop_event, state_q: queue.Queue):
    """
    Runs in a background thread.
    Pushes state dicts into state_q for the main thread to consume.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_buffer  = deque(maxlen=150)
    stable_hist   = deque(maxlen=STABLE_FRAMES_NEEDED)
    stage         = "idle"
    capture_start = None

    def push(extra=None):
        d = {"stage": stage}
        if extra:
            d.update(extra)
        try:
            state_q.put_nowait(d)
        except queue.Full:
            pass

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── STAGE: idle / detecting ────────────────────────────────────────
        if stage in ("idle", "detecting"):
            mask, corners = detector.run_seg(frame)

            if mask is None or corners is None:
                stable_hist.clear()
                stage = "idle"
                push({"frame": _encode_frame(frame), "corners": None,
                      "stable_count": 0, "area_ratio": 0.0,
                      "in_bounds": False, "sharpness": 0.0})
                continue

            h, w   = frame.shape[:2]
            fa     = h * w
            ma     = int(cv2.countNonZero(mask))
            area_r = ma / fa
            bounds_ok = all(0 <= x < w and 0 <= y < h for x, y in corners)
            area_ok   = MIN_AREA_RATIO <= area_r <= MAX_AREA_RATIO

            # Centroid stability
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx_ = M["m10"] / M["m00"]
                cy_ = M["m01"] / M["m00"]
                stable_hist.append((cx_, cy_))
            else:
                stable_hist.clear()

            stab = len(stable_hist)
            if stab >= STABLE_FRAMES_NEEDED:
                ref = stable_hist[0]
                stab_ok = all(
                    abs(p[0]-ref[0]) <= STABILITY_TOLERANCE and
                    abs(p[1]-ref[1]) <= STABILITY_TOLERANCE
                    for p in list(stable_hist)[1:]
                )
            else:
                stab_ok = False

            sharp = detector.calc_sharpness(frame, mask)
            stage = "detecting"

            push({"frame":        _encode_frame(draw_overlay(frame, corners, stage, stab)),
                  "corners":      corners.tolist(),
                  "area_ratio":   area_r,
                  "in_bounds":    bounds_ok,
                  "stable_count": stab,
                  "sharpness":    float(sharp)})

            if bounds_ok and area_ok and stab_ok:
                # Gate passed — start capture
                stage         = "capturing"
                capture_start = time.time()
                frame_buffer.clear()
                stable_hist.clear()
                detector.start_capture()

        # ── STAGE: capturing ───────────────────────────────────────────────
        elif stage == "capturing":
            elapsed  = time.time() - capture_start
            progress = min(elapsed / CAPTURE_DURATION, 1.0)

            mask, corners = detector.run_seg(frame)
            sharp = detector.calc_sharpness(frame, mask) if mask is not None else 0.0

            # Buffer frame
            frame_buffer.append({
                "frame":    frame.copy(),
                "mask":     mask,
                "corners":  corners,
                "sharpness":sharp,
            })

            overlay = draw_overlay(frame, corners, "capturing",
                                   STABLE_FRAMES_NEEDED)
            push({"frame":            _encode_frame(overlay),
                  "corners":          corners.tolist() if corners is not None else None,
                  "capture_progress": progress,
                  "sharpness":        float(sharp),
                  "area_ratio":       st.session_state.get("area_ratio", 0),
                  "in_bounds":        True,
                  "stable_count":     STABLE_FRAMES_NEEDED})

            if elapsed >= CAPTURE_DURATION:
                stage = "processing"

        # ── STAGE: processing ──────────────────────────────────────────────
        elif stage == "processing":
            # Pick sharpest frame — do NOT save the 2s video
            best = detector.get_sharpest_frame(frame_buffer)
            if best is None:
                stage = "idle"
                continue

            push({"frame":            _encode_frame(frame),
                  "stage":            "processing",
                  "capture_progress": 1.0,
                  "sharpness":        float(best["sharpness"]),
                  "area_ratio":       0.0,
                  "in_bounds":        True,
                  "stable_count":     STABLE_FRAMES_NEEDED,
                  "corners":          None})

            result = detector.process_sharpest_frame(best)

            # Encode warped frame for display
            warped_rgb = _encode_frame(result["warped_frame"])

            push({"stage":       "done",
                  "frame":       _encode_frame(result["frame"]),
                  "warped":      warped_rgb,
                  "result":      {
                      "id":          result["id"],
                      "timestamp":   result["timestamp"],
                      "ocr_result":  result["ocr_result"],
                      "sharpness":   result["sharpness"],
                      "seg_corners": result["seg_corners"],
                      "obb_corners": result["obb_corners"],
                  },
                  "sharpness":    float(result["sharpness"]),
                  "area_ratio":   0.0,
                  "in_bounds":    True,
                  "stable_count": STABLE_FRAMES_NEEDED,
                  "capture_progress": 1.0,
                  "corners":      None})

            # Pause briefly so the result is visible, then reset
            time.sleep(3.0)
            stage = "idle"
            frame_buffer.clear()
            push({"stage": "idle", "corners": None,
                  "area_ratio": 0.0, "in_bounds": False,
                  "stable_count": 0, "sharpness": 0.0,
                  "capture_progress": 0.0})

    cap.release()


# ──────────────────────────────────────────────────────────────────────────────
#  LOAD DETECTOR (cached so models load once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    from DefectDetectionOCR import DefectDetectionOCR
    from ultralytics import YOLO
    SEG_MODEL_PATH = "models/yolo11m-seg.pt"   # ← replace
    OBB_MODEL_PATH = "models/yolo11m-obb.pt"   # ← replace
    seg_model = YOLO(SEG_MODEL_PATH)
    obb_model = YOLO(OBB_MODEL_PATH)
    return DefectDetectionOCR(seg_model=seg_model, obb_model=obb_model)


# ──────────────────────────────────────────────────────────────────────────────
#  THREAD CONTROL
# ──────────────────────────────────────────────────────────────────────────────
def start_detection():
    detector  = load_detector()
    stop_evt  = threading.Event()
    state_q   = queue.Queue(maxsize=4)
    t = threading.Thread(
        target=detection_worker,
        args=(detector, stop_evt, state_q),
        daemon=True)
    t.start()
    st.session_state.update({
        "running":    True,
        "stop_event": stop_evt,
        "thread":     t,
        "state_q":    state_q,
    })

def stop_detection():
    if st.session_state.get("stop_event"):
        st.session_state["stop_event"].set()
    st.session_state.update({
        "running":    False,
        "stage":      "idle",
        "frame_rgb":  None,
        "warped_rgb": None,
    })

# ──────────────────────────────────────────────────────────────────────────────
#  CONSUME QUEUE (called on each rerun)
# ──────────────────────────────────────────────────────────────────────────────
def consume_queue():
    q = st.session_state.get("state_q")
    if q is None:
        return
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return

    st.session_state["stage"]            = latest.get("stage", st.session_state["stage"])
    st.session_state["stable_count"]     = latest.get("stable_count",     st.session_state["stable_count"])
    st.session_state["area_ratio"]       = latest.get("area_ratio",       st.session_state["area_ratio"])
    st.session_state["in_bounds"]        = latest.get("in_bounds",        st.session_state["in_bounds"])
    st.session_state["sharpness"]        = latest.get("sharpness",        st.session_state["sharpness"])
    st.session_state["capture_progress"] = latest.get("capture_progress", st.session_state["capture_progress"])

    if "frame" in latest:
        st.session_state["frame_rgb"] = latest["frame"]
    if "warped" in latest:
        st.session_state["warped_rgb"] = latest["warped"]
    if "result" in latest:
        st.session_state["latest_result"] = latest["result"]
        st.session_state["results"].append(latest["result"])

consume_queue()

# ──────────────────────────────────────────────────────────────────────────────
#  UI — HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="logo-bar">◉ DEFECTVISION</div>',
    unsafe_allow_html=True
)

stage   = st.session_state["stage"]
running = st.session_state["running"]

pill_map = {
    "idle":       ("STANDBY",    "pill-idle"),
    "detecting":  ("GATE CHECK", "pill-detect"),
    "capturing":  ("BUFFERING",  "pill-capture"),
    "processing": ("PROCESSING", "pill-process"),
    "done":       ("SCAN DONE ✓","pill-done"),
}
pill_label, pill_cls = pill_map.get(stage, ("STANDBY", "pill-idle"))

hcol1, hcol2, hcol3 = st.columns([5, 2, 2])
with hcol1:
    st.markdown(
        f'<span class="status-pill {pill_cls}">{pill_label}</span>',
        unsafe_allow_html=True
    )
with hcol2:
    if not running:
        if st.button("▶  START CAMERA"):
            start_detection()
            st.rerun()
    else:
        if st.button("⏹  STOP"):
            stop_detection()
            st.rerun()
with hcol3:
    if st.button("🗑  CLEAR RESULTS"):
        st.session_state["results"] = []
        st.session_state["latest_result"] = None
        st.rerun()

st.markdown('<hr style="border-color:#21262d; margin: 10px 0 16px;">', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  UI — MAIN LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

# ── LEFT: camera feed ─────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-header">LIVE FEED</div>', unsafe_allow_html=True)

    feed_slot = st.empty()

    if st.session_state["frame_rgb"] is not None:
        feed_slot.image(st.session_state["frame_rgb"], use_container_width=True,
                        caption="SEG overlay — real-time")
    else:
        feed_slot.markdown(
            """<div style="background:#111418;border:1px solid #21262d;border-radius:8px;
            height:360px;display:flex;align-items:center;justify-content:center;
            font-family:'Space Mono',monospace;color:#7d8590;font-size:13px;">
            CAMERA OFF — press START CAMERA</div>""",
            unsafe_allow_html=True
        )

    # Capture progress bar
    if stage == "capturing":
        prog = st.session_state["capture_progress"]
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:10px;'
            f'color:#ff6b35;letter-spacing:1px;margin-top:6px;">⏺ BUFFERING FRAMES</div>',
            unsafe_allow_html=True
        )
        st.progress(prog)

    # Processing spinner
    if stage == "processing":
        st.info("⚙️ Processing sharpest frame → OBB → OCR …")

    # Warped tag preview (shown during processing and done)
    if stage in ("processing", "done") and st.session_state["warped_rgb"] is not None:
        st.markdown('<div class="section-header" style="margin-top:14px;">WARPED TAG VIEW</div>',
                    unsafe_allow_html=True)
        st.image(st.session_state["warped_rgb"], use_container_width=True,
                 caption="Perspective-corrected tag")

# ── RIGHT: pipeline + policies + results ──────────────────────────────────────
with right:

    # ── Pipeline status ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">PIPELINE</div>', unsafe_allow_html=True)

    steps = [
        (["detecting","capturing","processing","done"], "pipe-dot-active",  "SEG Model",       "Running on every frame"),
        (["detecting","capturing","processing","done"], "pipe-dot-active",  "Policy Gate",     f"Bounds · Area · Stable ×{STABLE_FRAMES_NEEDED}"),
        (["capturing"],                                 "pipe-dot-capture", "2s Frame Buffer", "Recording frames (no video saved)"),
        (["processing","done"],                         "pipe-dot-process", "Sharpest Frame",  "Best focus selected"),
        (["processing","done"],                         "pipe-dot-process", "OBB Model",       "Oriented bounding box"),
        (["processing","done"],                         "pipe-dot-process", "OCR",             "EasyOCR → text extraction"),
    ]

    for active, dot_cls, name, desc in steps:
        cls = dot_cls if stage in active else "pipe-dot"
        st.markdown(f"""
            <div class="pipe-row">
                <div class="pipe-dot {cls}"></div>
                <div>
                    <span style="font-weight:600">{name}</span>
                    <span style="color:#7d8590;font-size:11px;margin-left:8px;">{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#21262d; margin:10px 0;">', unsafe_allow_html=True)

    # ── Gate policies ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">GATE POLICIES</div>', unsafe_allow_html=True)

    stab = st.session_state["stable_count"]
    stab_pct = stab / STABLE_FRAMES_NEEDED

    area_r = st.session_state["area_ratio"]
    area_ok = MIN_AREA_RATIO <= area_r <= MAX_AREA_RATIO
    area_pct = min(area_r / MAX_AREA_RATIO, 1.0)

    sharp   = st.session_state["sharpness"]
    sharp_pct = min(sharp / 800.0, 1.0)

    bounds  = st.session_state["in_bounds"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="color:{'#00e5a0' if bounds else '#ef4444'}">
                {'✓' if bounds else '✗'}</div>
            <div class="metric-label">IN BOUNDS</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="color:{'#00e5a0' if area_ok else '#f59e0b'}">
                {area_r*100:.0f}%</div>
            <div class="metric-label">AREA</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="color:{'#00e5a0' if stab>=STABLE_FRAMES_NEEDED else '#f59e0b'}">
                {stab}/{STABLE_FRAMES_NEEDED}</div>
            <div class="metric-label">STABLE</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="color:#00b4d8">{sharp:.0f}</div>
            <div class="metric-label">SHARPNESS</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.caption("Stability")
    st.progress(stab_pct)
    st.caption("Area ratio")
    st.progress(area_pct)
    st.caption("Sharpness")
    st.progress(sharp_pct)

    st.markdown('<hr style="border-color:#21262d; margin:10px 0;">', unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">RESULTS '
        f'<span style="color:#00e5a0">{len(st.session_state["results"])} scans</span></div>',
        unsafe_allow_html=True
    )

    results = st.session_state["results"]
    if not results:
        st.markdown(
            '<div style="color:#7d8590;font-family:Space Mono,monospace;'
            'font-size:12px;padding:20px 0;text-align:center;">'
            '📭 No scans yet<br><small>Point a tag at the camera</small></div>',
            unsafe_allow_html=True
        )
    else:
        for r in reversed(results[-10:]):   # show latest 10
            ts = datetime.fromisoformat(r["timestamp"]).strftime("%H:%M:%S")
            st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-family:Space Mono,monospace;font-size:11px;
                              color:#00e5a0;font-weight:700;">#{r['id']:04d}</span>
                        <span style="font-family:Space Mono,monospace;font-size:10px;
                              color:#7d8590;">{ts}</span>
                    </div>
                    <div class="result-ocr">{r['ocr_result'] or '— no text detected —'}</div>
                    <div style="font-family:Space Mono,monospace;font-size:10px;color:#7d8590;">
                        sharpness: {float(r['sharpness']):.1f} &nbsp;|&nbsp;
                        obb: {'✓' if r.get('obb_corners') else '—'}
                    </div>
                </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  AUTO-RERUN while detection is running
# ──────────────────────────────────────────────────────────────────────────────
if running:
    time.sleep(0.08)   # ~12 fps UI refresh
    st.rerun()