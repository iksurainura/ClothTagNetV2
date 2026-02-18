"""
frontendOCR.py  —  Streamlit frontend for DefectDetectionOCR
Run with:  streamlit run frontendOCR.py
DefectDetectionOCR.py must be in the same directory.

Correct pipeline
────────────────
Frontend START button
  → SEG model runs on every frame
  → Gate: tag fully inside frame + stable N frames
  → 2-second video capture
  → Sharpest frame selected
  → Perspective warp (corners from SEG mask)
  → OBB model on warped image
  → OCR on warped image
  → Results displayed
"""

import streamlit as st
import cv2
import numpy as np
import threading
import time
from datetime import datetime

from DefectDetectionOCR import DefectDetectionOCR

# ══════════════════════════════════════════════════════════════════════════ #
#  ✏️  EDIT THESE — point to your local .pt files                           #
# ══════════════════════════════════════════════════════════════════════════ #
SEG_MODEL_PATH = r"model\seg_model.pt"
OBB_MODEL_PATH = r"model\obb_model.pt"

# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE CONFIG                                                               #
# ══════════════════════════════════════════════════════════════════════════ #
st.set_page_config(
    page_title="Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════ #
#  CSS                                                                       #
# ══════════════════════════════════════════════════════════════════════════ #
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

    /* ── page title ── */
    .page-title {
        font-size: 26px; font-weight: 700; color: #58a6ff;
        padding: 14px 0 4px; border-bottom: 1px solid #21262d;
        margin-bottom: 16px;
    }

    /* ── metric card ── */
    .metric-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px 18px; text-align: center;
    }
    .metric-label {
        font-size: 11px; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #58a6ff; margin-top: 4px; }

    /* ── status badge ── */
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px;
    }
    .badge-green  { background:#0d2a1a; color:#3fb950; border:1px solid #3fb950; }
    .badge-yellow { background:#2a2200; color:#d29922; border:1px solid #d29922; }
    .badge-blue   { background:#0d1f33; color:#58a6ff; border:1px solid #58a6ff; }
    .badge-red    { background:#2a0d0d; color:#f85149; border:1px solid #f85149; }

    /* ── section header ── */
    .sec-hdr {
        font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 5px; margin: 14px 0 10px;
    }

    /* ── gate log box ── */
    .gate-log {
        background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 12px; font-family: monospace; font-size: 11px;
        color: #8b949e; height: 220px; overflow-y: auto;
    }

    /* ── result card ── */
    .result-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px; margin-bottom: 14px;
    }
    .result-hdr {
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 10px; border-bottom: 1px solid #21262d; margin-bottom: 12px;
    }
    .result-id { font-size: 16px; font-weight: 700; color: #58a6ff; }
    .result-ts { font-size: 11px; color: #8b949e; }

    /* ── OCR box ── */
    .ocr-box {
        background: #0d1117; border-left: 3px solid #58a6ff;
        border-radius: 0 6px 6px 0; padding: 10px 12px;
        font-family: monospace; font-size: 13px; color: #e0e0e0;
        word-break: break-all; min-height: 40px;
    }

    /* ── pipeline step labels ── */
    .pipe-step {
        display: inline-block; background: #21262d;
        border-radius: 4px; padding: 2px 8px;
        font-size: 11px; color: #8b949e; margin-right: 4px;
    }
    .pipe-arrow { color: #3fb950; font-weight: 700; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════ #
#  THREAD-SAFE SHARED STATE  (camera thread → main thread)                  #
# ══════════════════════════════════════════════════════════════════════════ #
_lock = threading.Lock()
_shared = {
    "live_frame"  : None,       # latest raw frame (RGB np.ndarray)
    "seg_overlay" : None,       # frame with seg mask drawn (RGB np.ndarray)
    "gate_status" : "idle",     # idle | waiting | capturing | processing
    "gate_log"    : [],
    "frame_count" : 0,
    "capture_pct" : 0.0,        # 0.0–1.0 capture progress
}

def _set(k, v):
    with _lock: _shared[k] = v

def _get(k):
    with _lock: return _shared[k]

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        _shared["gate_log"].append(f"[{ts}] {msg}")
        _shared["gate_log"] = _shared["gate_log"][-50:]

def _inc():
    with _lock: _shared["frame_count"] += 1


# ══════════════════════════════════════════════════════════════════════════ #
#  SESSION STATE INIT                                                        #
# ══════════════════════════════════════════════════════════════════════════ #
for _k, _v in {
    "detector"      : None,
    "running"       : False,
    "stop_event"    : threading.Event(),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════ #
#  MODEL LOADING                                                             #
# ══════════════════════════════════════════════════════════════════════════ #
@st.cache_resource
def load_models(seg_path, obb_path):
    from ultralytics import YOLO
    return YOLO(seg_path), YOLO(obb_path)


# ══════════════════════════════════════════════════════════════════════════ #
#  DRAW SEG OVERLAY  (green mask + corner dots on frame)                    #
# ══════════════════════════════════════════════════════════════════════════ #
def draw_seg_overlay(frame: np.ndarray, mask: np.ndarray,
                     corners: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    if mask is not None:
        green = np.zeros_like(vis)
        green[:, :, 1] = mask           # green channel = mask
        vis = cv2.addWeighted(vis, 1.0, green, 0.35, 0)
    if corners is not None:
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 80), 2)
        for pt in pts:
            cv2.circle(vis, tuple(pt), 5, (0, 255, 80), -1)
    return vis


# ══════════════════════════════════════════════════════════════════════════ #
#  INFERENCE LOOP  (separate thread — keeps live feed unblocked)            #
# ══════════════════════════════════════════════════════════════════════════ #
def inference_loop(detector: DefectDetectionOCR,
                   frame_queue,
                   stop_event: threading.Event):
    """
    Pulls latest frame from queue and runs seg inference.
    Completely decoupled from camera read loop so live feed never stutters.
    """
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        # ── PHASE 1 — Gate ───────────────────────────────────────────────
        if not detector.is_capturing:
            _set("gate_status", "waiting")
            passed, mask, corners = detector.check_tag_policy(frame)

            overlay = draw_seg_overlay(frame, mask, corners)
            _set("seg_overlay", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            if passed:
                _log("✅ Tag policy passed — starting capture")
                detector.start_capture()
                _set("gate_status", "capturing")
            else:
                if mask is not None:
                    _log("⏳ Tag detected — waiting for stability...")

        # ── PHASE 2 — Capture ────────────────────────────────────────────
        else:
            _set("gate_status", "capturing")
            mask, corners = detector.run_seg(frame)
            sharpness     = detector.calc_sharpness(frame, mask)

            detector.frame_buffer.append({
                "frame"    : frame.copy(),
                "mask"     : mask,
                "corners"  : corners,
                "sharpness": sharpness,
            })

            elapsed = (datetime.now() - detector.capture_start_time).total_seconds()
            _set("capture_pct", min(elapsed / detector.capture_duration, 1.0))

            if detector.should_stop_capture():
                detector.is_capturing = False
                _set("gate_status", "processing")
                _log("📸 Capture done — selecting sharpest frame...")

                best = detector.get_sharpest_frame()
                if best:
                    _log(f"🔍 Sharpest frame score: {best['sharpness']:.1f}")
                    _log("🔄 Warp → OBB → OCR...")
                    result = detector.process_sharpest_frame(best)
                    _log(f"✅ Done! OCR: '{result['ocr_result']}'")
                else:
                    _log("⚠️ No valid frames in buffer")

                _set("capture_pct", 0.0)
                _set("gate_status", "waiting")
                _log("👁 Watching for next tag...")


# ══════════════════════════════════════════════════════════════════════════ #
#  CAMERA LOOP  (background thread — ONLY reads frames, no inference here)  #
# ══════════════════════════════════════════════════════════════════════════ #
def camera_loop(detector: DefectDetectionOCR,
                cam_idx: int,
                stop_event: threading.Event):
    """
    Reads camera at full fps and pushes frames to shared state instantly.
    Inference runs in a separate thread so the live feed is NEVER blocked.
    """
    import queue as _queue
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # maxsize=1: inference always gets latest frame, never queues stale ones
    frame_q = _queue.Queue(maxsize=1)

    inf_thread = threading.Thread(
        target=inference_loop,
        args=(detector, frame_q, stop_event),
        daemon=True,
    )
    inf_thread.start()

    _set("gate_status", "waiting")
    _log("Camera started — watching for tag...")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        _inc()

        # Live feed: always at full camera fps, no inference blocking it
        _set("live_frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Push to inference queue — drop stale frame if inference is busy
        try:
            frame_q.get_nowait()
        except Exception:
            pass
        try:
            frame_q.put_nowait(frame.copy())
        except Exception:
            pass

    cap.release()
    _set("gate_status", "idle")
    _log("Camera stopped.")


# ══════════════════════════════════════════════════════════════════════════ #
#  HELPERS                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #
def badge(text, color="blue"):
    cls = {"green":"badge-green","yellow":"badge-yellow",
           "blue":"badge-blue","red":"badge-red"}.get(color,"badge-blue")
    return f'<span class="badge {cls}">{text}</span>'

def status_badge():
    s = _get("gate_status")
    m = {
        "idle"       : ("⏸ Idle",        "yellow"),
        "waiting"    : ("👁 Watching",    "blue"),
        "capturing"  : ("🎬 Capturing",   "green"),
        "processing" : ("⚙️ Processing",  "yellow"),
    }
    txt, col = m.get(s, ("Unknown", "red"))
    return badge(txt, col)

def metric(label, value):
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>')


# ══════════════════════════════════════════════════════════════════════════ #
#  SIDEBAR                                                                   #
# ══════════════════════════════════════════════════════════════════════════ #
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    cam_idx = st.number_input("Camera index", 0, 10, 0)
    cap_dur = st.slider("Capture duration (s)", 1, 10, 2)

    st.markdown("---")
    st.markdown("### Gate Policies")
    stable_n   = st.slider("Stable frames needed",    2, 15,  5)
    stable_tol = st.slider("Centroid tolerance (px)", 5, 50, 15)
    min_area   = st.slider("Min tag area (%)",         1, 30,  2) / 100
    max_area   = st.slider("Max tag area (%)",        30, 99, 90) / 100

    # Push to module constants live
    import DefectDetectionOCR as _dd
    _dd.STABLE_FRAMES_NEEDED = stable_n
    _dd.STABILITY_TOLERANCE  = stable_tol
    _dd.MIN_AREA_RATIO       = min_area
    _dd.MAX_AREA_RATIO       = max_area

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with c2:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)
    st.markdown("---")
    clear_btn = st.button("🗑 Clear Results", use_container_width=True)

    # Pipeline reminder
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        '<span class="pipe-step">SEG</span>'
        '<span class="pipe-arrow">→</span>'
        '<span class="pipe-step">Gate</span>'
        '<span class="pipe-arrow">→</span>'
        '<span class="pipe-step">Capture</span>'
        '<span class="pipe-arrow">→</span>'
        '<span class="pipe-step">Warp</span>'
        '<span class="pipe-arrow">→</span>'
        '<span class="pipe-step">OBB</span>'
        '<span class="pipe-arrow">→</span>'
        '<span class="pipe-step">OCR</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  BUTTON ACTIONS                                                            #
# ══════════════════════════════════════════════════════════════════════════ #
if start_btn and not st.session_state["running"]:
    try:
        seg_model, obb_model = load_models(SEG_MODEL_PATH, OBB_MODEL_PATH)
        detector = DefectDetectionOCR(
            seg_model=seg_model,
            obb_model=obb_model,
            capture_duration=cap_dur,
        )
        st.session_state["detector"]   = detector
        st.session_state["running"]    = True
        st.session_state["stop_event"] = threading.Event()

        # Reset shared state
        _set("gate_log",    [])
        _set("frame_count", 0)
        _set("gate_status", "waiting")
        _set("live_frame",  None)
        _set("seg_overlay", None)
        _set("capture_pct", 0.0)

        t = threading.Thread(
            target=camera_loop,
            args=(detector, int(cam_idx), st.session_state["stop_event"]),
            daemon=True,
        )
        t.start()
        st.success("✅ System started!")
    except Exception as e:
        st.error(f"Failed to start: {e}")

if stop_btn and st.session_state["running"]:
    st.session_state["stop_event"].set()
    st.session_state["running"] = False
    st.warning("⏹ System stopped.")

if clear_btn and st.session_state["detector"] is not None:
    st.session_state["detector"].clear_results()
    st.success("Results cleared.")


# ══════════════════════════════════════════════════════════════════════════ #
#  MAIN LAYOUT                                                               #
# ══════════════════════════════════════════════════════════════════════════ #
st.markdown('<div class="page-title">🔍 Defect Detection Dashboard</div>',
            unsafe_allow_html=True)

detector: DefectDetectionOCR = st.session_state.get("detector")
all_results = detector.get_all_results() if detector else []

# ── Metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric("Frames", _get("frame_count")), unsafe_allow_html=True)
with m2:
    st.markdown(metric("Tags Processed", len(all_results)), unsafe_allow_html=True)
with m3:
    avg = np.mean([r["sharpness"] for r in all_results]) if all_results else 0
    st.markdown(metric("Avg Sharpness", f"{avg:.0f}"), unsafe_allow_html=True)
with m4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Status</div>'
        f'<div style="margin-top:8px">{status_badge()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Capture progress bar (only visible while capturing) ───────────────────
cap_pct = _get("capture_pct")
if cap_pct > 0:
    st.progress(cap_pct, text=f"Capturing... {cap_pct*100:.0f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Live feed + Gate log ──────────────────────────────────────────────────
feed_col, log_col = st.columns([3, 2])

with feed_col:
    st.markdown('<div class="sec-hdr">📷 Live Feed  (with SEG overlay)</div>',
                unsafe_allow_html=True)
    feed_placeholder = st.empty()

    # Prefer overlay frame (shows seg mask); fall back to raw frame
    overlay = _get("seg_overlay")
    raw     = _get("live_frame")
    display = overlay if overlay is not None else raw

    if display is not None:
        feed_placeholder.image(display, use_column_width=True)
    else:
        feed_placeholder.info("Press ▶ Start to begin.")

with log_col:
    st.markdown('<div class="sec-hdr">📋 Pipeline Log</div>',
                unsafe_allow_html=True)
    logs = _get("gate_log")
    log_html = "<br>".join(reversed(logs[-30:])) if logs else "No events yet."
    st.markdown(f'<div class="gate-log">{log_html}</div>', unsafe_allow_html=True)

    if all_results:
        st.markdown('<div class="sec-hdr">📊 Stats</div>', unsafe_allow_html=True)
        sharp_scores = [r["sharpness"] for r in all_results]
        st.markdown(f"**Best:** `{max(sharp_scores):.1f}` &nbsp; "
                    f"**Worst:** `{min(sharp_scores):.1f}`")
        last_ocr = all_results[-1]["ocr_result"]
        st.markdown(f"**Last OCR:** `{last_ocr or 'none'}`")

st.markdown("---")

# ── Results gallery ───────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🗂️ Results</div>', unsafe_allow_html=True)

if not all_results:
    st.info("No results yet — system will process a tag automatically once gate passes.")
else:
    for result in reversed(all_results):
        r_id    = result["id"]
        r_ts    = result["timestamp"]
        r_sharp = result["sharpness"]
        r_badge = badge(f"Sharpness {r_sharp:.0f}", "blue")

        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-hdr">'
            f'  <span class="result-id">Tag #{r_id}</span>'
            f'  <span class="result-ts">{r_ts}</span>'
            f'  {r_badge}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        img_col, info_col = st.columns([2, 3])

        with img_col:
            if result.get("warped_frame") is not None:
                warped_rgb = cv2.cvtColor(result["warped_frame"], cv2.COLOR_BGR2RGB)
                st.image(warped_rgb, caption="Perspective-corrected tag",
                         use_column_width=True)
            with st.expander("Original frame"):
                if result.get("frame") is not None:
                    orig_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, use_column_width=True)

        with info_col:
            st.markdown("**OCR Output**")
            ocr = result.get("ocr_result") or "_(no text detected)_"
            st.markdown(f'<div class="ocr-box">{ocr}</div>',
                        unsafe_allow_html=True)

            st.markdown("**SEG Corners (warp source)**")
            seg_c = result.get("seg_corners")
            if seg_c:
                cols = st.columns(2)
                for i, (x, y) in enumerate(seg_c):
                    cols[i % 2].markdown(f"`P{i+1}` x:`{x:.0f}` y:`{y:.0f}`")
            else:
                st.markdown("_fallback crop used_")

            st.markdown("**OBB Result (on warped)**")
            obb_c = result.get("obb_corners")
            if obb_c:
                st.markdown(f"`{len(obb_c)} corners detected on warped image`")
            else:
                st.markdown("_no OBB detection on warped image_")

            st.markdown("**Sharpness**")
            st.progress(min(r_sharp / 5000, 1.0), text=f"{r_sharp:.1f}")

        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════ #
#  AUTO-REFRESH                                                              #
# ══════════════════════════════════════════════════════════════════════════ #
if st.session_state["running"]:
    time.sleep(0.4)
    st.rerun()