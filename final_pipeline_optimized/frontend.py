"""
frontendOCR.py  —  Streamlit frontend for DefectDetection
Run with:  streamlit run frontendOCR.py
DefectDetection.py must be in the same directory.

Pipeline
────────
START → SEG gate (every Nth frame, FP16) → capture buffer → sharpest frame
      → perspective warp → OBB (FP16) → OCR → results

Live feed
─────────
Uses YOLO result.plot() — same approach as reference app.py snippet.
Shows masks + boxes + labels on every frame exactly as YOLO renders them.
Raw frame shows while inference runs; annotated frame replaces it as soon
as inference completes. Both threads are fully decoupled so the feed never
stutters regardless of inference speed.
"""

import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from datetime import datetime
from DefectDetection import DefectDetection


# ══════════════════════════════════════════════════════════════════════════ #
#  ✏️  EDIT THESE — point to your trained .pt files                         #
# ══════════════════════════════════════════════════════════════════════════ #
SEG_MODEL_PATH = "models/yolo11m-seg.pt"
OBB_MODEL_PATH = "models/yolo11m-obb.pt"


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
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem; }
    header[data-testid="stHeader"] { display: none; }
    #MainMenu, footer { visibility: hidden; }

    .stApp { background: #0d1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
    .page-title {
        font-size: 26px; font-weight: 700; color: #58a6ff;
        padding: 14px 0 4px; border-bottom: 1px solid #21262d;
        margin-bottom: 16px;
    }
    .metric-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px 18px; text-align: center;
    }
    .metric-label { font-size: 11px; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 700;
        color: #58a6ff; margin-top: 4px; }
    .badge { display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px; }
    .badge-green  { background:#0d2a1a; color:#3fb950; border:1px solid #3fb950; }
    .badge-yellow { background:#2a2200; color:#d29922; border:1px solid #d29922; }
    .badge-blue   { background:#0d1f33; color:#58a6ff; border:1px solid #58a6ff; }
    .badge-red    { background:#2a0d0d; color:#f85149; border:1px solid #f85149; }
    .sec-hdr { font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #21262d; padding-bottom: 5px; margin: 14px 0 10px; }
    .gate-log { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 12px; font-family: monospace; font-size: 11px;
        color: #8b949e; height: 220px; overflow-y: auto; }
    .result-card { background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px; margin-bottom: 14px; }
    .result-hdr { display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 10px; border-bottom: 1px solid #21262d; margin-bottom: 12px; }
    .result-id { font-size: 16px; font-weight: 700; color: #58a6ff; }
    .result-ts { font-size: 11px; color: #8b949e; }
    .ocr-box { background: #0d1117; border-left: 3px solid #58a6ff;
        border-radius: 0 6px 6px 0; padding: 10px 12px;
        font-family: monospace; font-size: 13px; color: #e0e0e0;
        word-break: break-all; min-height: 40px; }
    .gpu-chip { display: inline-block; background: #1a2a1a;
        border: 1px solid #3fb950; border-radius: 4px;
        padding: 2px 8px; font-size: 11px; color: #3fb950; font-family: monospace; }
    .pipe-step { display: inline-block; background: #21262d;
        border-radius: 4px; padding: 2px 8px;
        font-size: 11px; color: #8b949e; margin-right: 4px; }
    .pipe-arrow { color: #3fb950; font-weight: 700; margin-right: 4px; }
    .feed-label { font-size: 10px; color: #8b949e; font-family: monospace;
        text-align: right; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════ #
#  THREAD-SAFE SHARED STATE                                                  #
#  Camera thread (Thread 1) and inference thread (Thread 2) write here.    #
#  Main Streamlit thread reads here on every rerun.                         #
#  NEVER touch st.session_state from background threads.                   #
# ══════════════════════════════════════════════════════════════════════════ #
_lock   = threading.Lock()
_shared = {
    "live_frame"    : None,   # RGB — raw frame, updated at full camera fps
    "plotted_frame" : None,   # RGB — YOLO .plot() output, updated per inference
    "gate_status"   : "idle",
    "gate_log"      : [],
    "frame_count"   : 0,
    "capture_pct"   : 0.0,
    "inf_fps"       : 0.0,
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
#  SESSION STATE INIT  (main thread only)                                   #
# ══════════════════════════════════════════════════════════════════════════ #
for _k, _v in {
    "detector"  : None,
    "running"   : False,
    "stop_event": threading.Event(),
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════ #
#  MODEL LOADING  (cached — loads once per session)                         #
# ══════════════════════════════════════════════════════════════════════════ #
@st.cache_resource
def load_models(seg_path, obb_path):
    from ultralytics import YOLO
    return YOLO(seg_path), YOLO(obb_path)


# ══════════════════════════════════════════════════════════════════════════ #
#  LIVE FEED RENDERING                                                       #
#                                                                            #
#  Inspired by reference app.py:                                             #
#    live_view.image(cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB))       #
#                                                                            #
#  seg_result.plot() — YOLO built-in renderer, draws:                       #
#    • Segmentation masks (coloured fill)                                    #
#    • Bounding boxes                                                        #
#    • Class labels + confidence scores                                      #
#  Then we draw the 4 warp-corner dots on top in bright green.              #
# ══════════════════════════════════════════════════════════════════════════ #
def make_plotted_frame(seg_result, corners: np.ndarray = None) -> np.ndarray:
    """
    Call seg_result.plot() to get YOLO's annotated BGR frame.
    Optionally overlay the 4 rotated-bbox corner dots (bright green).
    Return RGB for Streamlit.
    """
    annotated = seg_result.plot()          # YOLO draws masks + boxes + labels

    if corners is not None:
        pts = corners.astype(np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 80), thickness=2)
        for pt in pts:
            cv2.circle(annotated, tuple(pt), radius=7,
                       color=(0, 255, 80), thickness=-1)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════ #
#  INFERENCE LOOP  (Thread 2)                                                #
#  Runs seg → gate/capture logic → updates plotted_frame.                   #
#  Completely decoupled from camera read so live feed never stutters.       #
# ══════════════════════════════════════════════════════════════════════════ #
def inference_loop(detector: DefectDetection,
                   frame_queue: queue.Queue,
                   stop_event: threading.Event):

    inf_times = deque(maxlen=10)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        t0 = time.perf_counter()

        # ── Gate phase ────────────────────────────────────────────────────
        if not detector.is_capturing:
            _set("gate_status", "waiting")

            # check_tag_policy always returns 4 values
            passed, mask, corners, seg_result = detector.check_tag_policy(frame)

            # Update plotted frame whenever we have a fresh inference result
            # (seg_result is None on skipped frames — keep previous overlay)
            if seg_result is not None:
                _set("plotted_frame", make_plotted_frame(seg_result, corners))

            if passed:
                _log("✅ Gate passed — starting capture")
                detector.start_capture()
                _set("gate_status", "capturing")
            elif mask is not None:
                _log("⏳ Tag visible — waiting for stability...")

        # ── Capture phase ─────────────────────────────────────────────────
        else:
            _set("gate_status", "capturing")

            # run_seg always returns 3 values
            mask, corners, seg_result = detector.run_seg(frame)
            sharpness = detector.calc_sharpness(frame, mask)

            # Keep live view annotated during capture too
            if seg_result is not None:
                _set("plotted_frame", make_plotted_frame(seg_result, corners))

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
                _log("📸 Capture done — finding sharpest frame...")

                best = detector.get_sharpest_frame()
                if best:
                    _log(f"🔍 Sharpest score: {best['sharpness']:.1f}")
                    _log("⚙️  Warp → OBB → OCR running...")
                    result = detector.process_sharpest_frame(best)
                    _log(f"✅ Done! OCR: '{result['ocr_result']}'")
                else:
                    _log("⚠️ No valid frames captured")

                _set("capture_pct", 0.0)
                _set("gate_status", "waiting")
                _log("👁 Watching for next tag...")

        # Track inference fps
        inf_times.append(time.perf_counter() - t0)
        if len(inf_times) > 1:
            avg_t = sum(inf_times) / len(inf_times)
            _set("inf_fps", round(1.0 / avg_t, 1) if avg_t > 0 else 0)


# ══════════════════════════════════════════════════════════════════════════ #
#  CAMERA LOOP  (Thread 1)                                                   #
#  ONLY reads frames and pushes to shared state + inference queue.          #
#  Zero inference here — live feed is always at full camera fps.            #
# ══════════════════════════════════════════════════════════════════════════ #
def camera_loop(detector: DefectDetection,
                cam_idx: int,
                stop_event: threading.Event):

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # always freshest frame

    # maxsize=1 — if inference is slow, old frame is dropped, newest is used
    frame_q = queue.Queue(maxsize=1)

    # Start inference thread
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

        # Raw frame → always at full camera fps, never blocked by inference
        _set("live_frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Push to inference queue — drop stale if inference still busy
        try:
            frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    cap.release()
    _set("gate_status", "idle")
    _log("Camera stopped.")


# ══════════════════════════════════════════════════════════════════════════ #
#  UI HELPERS                                                                #
# ══════════════════════════════════════════════════════════════════════════ #
def badge(text, color="blue"):
    cls = {"green": "badge-green", "yellow": "badge-yellow",
           "blue": "badge-blue", "red": "badge-red"}.get(color, "badge-blue")
    return f'<span class="badge {cls}">{text}</span>'

def status_badge():
    s = _get("gate_status")
    m = {
        "idle"      : ("⏸ Idle",        "yellow"),
        "waiting"   : ("👁 Watching",    "blue"),
        "capturing" : ("🎬 Capturing",   "green"),
        "processing": ("⚙️ Processing",  "yellow"),
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

    import torch
    gpu_ok     = torch.cuda.is_available()
    gpu_name   = torch.cuda.get_device_name(0) if gpu_ok else "Not available"
    chip_color = "#3fb950" if gpu_ok else "#f85149"
    gpu_label  = ("✓ " + gpu_name) if gpu_ok else "✗ CPU only"
    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'<span style="font-size:11px;color:#8b949e">GPU &nbsp;</span>'
        f'<span class="gpu-chip" style="border-color:{chip_color};color:{chip_color}">'
        f'{gpu_label}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    cam_idx = st.number_input("Camera index", 0, 10, 0)
    cap_dur = st.slider("Capture duration (s)", 1, 10, 2)

    st.markdown("---")
    st.markdown("### Gate Policies")
    stable_n   = st.slider("Stable frames needed",      2, 15,  5)
    stable_tol = st.slider("Centroid tolerance (px)",   5, 50, 15)
    min_area   = st.slider("Min tag area (%)",           1, 30,  2) / 100
    max_area   = st.slider("Max tag area (%)",          30, 99, 90) / 100
    skip_n     = st.slider("Gate frame skip (every N)", 1, 10,  3)
    seg_conf   = st.slider("SEG confidence threshold", 0.1, 1.0, 0.55, 0.05)

    import DefectDetection as _dd
    _dd.STABLE_FRAMES_NEEDED = stable_n
    _dd.STABILITY_TOLERANCE  = stable_tol
    _dd.MIN_AREA_RATIO       = min_area
    _dd.MAX_AREA_RATIO       = max_area
    _dd.GATE_SKIP_FRAMES     = skip_n
    _dd.SEG_CONF_THRESHOLD   = seg_conf

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with c2:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)
    st.markdown("---")
    clear_btn = st.button("🗑 Clear Results", use_container_width=True)

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        '<span class="pipe-step">SEG↓N</span>'
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
    st.markdown(
        '<div style="font-size:10px;color:#8b949e;margin-top:6px">'
        'FP16 · no_grad · imgsz=640 · buf=1 · warmup · EasyOCR cached</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  BUTTON ACTIONS                                                            #
# ══════════════════════════════════════════════════════════════════════════ #
if start_btn and not st.session_state["running"]:
    try:
        seg_model, obb_model = load_models(SEG_MODEL_PATH, OBB_MODEL_PATH)

        # __init__ runs warmup automatically
        detector = DefectDetection(
            seg_model=seg_model,
            obb_model=obb_model,
            capture_duration=cap_dur,
        )
        st.session_state["detector"]   = detector
        st.session_state["running"]    = True
        st.session_state["stop_event"] = threading.Event()

        _set("gate_log",      [])
        _set("frame_count",   0)
        _set("gate_status",   "waiting")
        _set("live_frame",    None)
        _set("plotted_frame", None)
        _set("capture_pct",   0.0)

        t = threading.Thread(
            target=camera_loop,
            args=(detector, int(cam_idx), st.session_state["stop_event"]),
            daemon=True,
        )
        t.start()

        gpu_status = "ON (FP16)" if gpu_ok else "OFF (CPU only)"
        st.success(f"✅ Started! GPU: {gpu_status}")
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

detector: DefectDetection = st.session_state.get("detector")
all_results = detector.get_all_results() if detector else []

# ── Metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(metric("Frames", _get("frame_count")), unsafe_allow_html=True)
with m2:
    st.markdown(metric("Tags", len(all_results)), unsafe_allow_html=True)
with m3:
    avg = np.mean([r["sharpness"] for r in all_results]) if all_results else 0
    st.markdown(metric("Avg Sharp", f"{avg:.0f}"), unsafe_allow_html=True)
with m4:
    st.markdown(metric("Inf FPS", f"{_get('inf_fps'):.1f}"), unsafe_allow_html=True)
with m5:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Status</div>'
        f'<div style="margin-top:8px">{status_badge()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Capture progress ──────────────────────────────────────────────────────
cap_pct = _get("capture_pct")
if cap_pct > 0:
    st.progress(cap_pct, text=f"Capturing... {cap_pct*100:.0f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Live feed + Log ───────────────────────────────────────────────────────
feed_col, log_col = st.columns([3, 2])

with feed_col:
    st.markdown('<div class="sec-hdr">📷 Live Feed</div>', unsafe_allow_html=True)

    # Display logic — mirrors reference app.py:
    #   live_view.image(cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB))
    #
    # Priority: plotted_frame (YOLO .plot() annotated) > raw live_frame
    # plotted_frame updates at inference speed (every ~0.5-1.5s on CPU)
    # live_frame updates at full camera fps (30fps) — shown while inference runs
    plotted = _get("plotted_frame")
    raw     = _get("live_frame")
    display = plotted if plotted is not None else raw

    if display is not None:
        st.image(display, use_column_width=True)
        # Label shows which feed is active
        label = "🟢 YOLO annotated (masks + boxes)" if plotted is not None \
                else "⚪ Raw feed (inference not started)"
        st.markdown(f'<div class="feed-label">{label}</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Press ▶ Start to begin.")

with log_col:
    st.markdown('<div class="sec-hdr">📋 Pipeline Log</div>',
                unsafe_allow_html=True)
    logs     = _get("gate_log")
    log_html = "<br>".join(reversed(logs[-30:])) if logs else "No events yet."
    st.markdown(f'<div class="gate-log">{log_html}</div>', unsafe_allow_html=True)

    if all_results:
        st.markdown('<div class="sec-hdr">📊 Stats</div>', unsafe_allow_html=True)
        scores = [r["sharpness"] for r in all_results]
        st.markdown(
            f"**Best:** `{max(scores):.1f}` &nbsp; **Worst:** `{min(scores):.1f}`"
        )
        last_ocr = all_results[-1]["ocr_result"]
        st.markdown(f"**Last OCR:** `{last_ocr or 'none'}`")

st.markdown("---")

# ── Results gallery ───────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🗂️ Results</div>', unsafe_allow_html=True)

if not all_results:
    st.info("No results yet — system processes a tag automatically when gate passes.")
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
            # Primary: OBB-annotated warped frame (.plot() output)
            # Fallback: plain warped frame
            display_img = result.get("obb_annotated") or result.get("warped_frame")
            if display_img is not None:
                caption = "Warped + OBB detections" \
                          if result.get("obb_annotated") is not None \
                          else "Perspective-corrected tag"
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                         caption=caption, use_column_width=True)

            tab1, tab2 = st.tabs(["Plain warped", "Original frame"])
            with tab1:
                if result.get("warped_frame") is not None:
                    st.image(
                        cv2.cvtColor(result["warped_frame"], cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                    )
            with tab2:
                if result.get("frame") is not None:
                    st.image(
                        cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                    )

        with info_col:
            st.markdown("**OCR Output**")
            ocr = result.get("ocr_result") or "_(no text detected)_"
            st.markdown(f'<div class="ocr-box">{ocr}</div>',
                        unsafe_allow_html=True)

            st.markdown("**SEG Corners**")
            seg_c = result.get("seg_corners")
            if seg_c:
                cols = st.columns(2)
                for i, (x, y) in enumerate(seg_c):
                    cols[i % 2].markdown(f"`P{i+1}` x:`{x:.0f}` y:`{y:.0f}`")
            else:
                st.markdown("_fallback crop used_")

            st.markdown("**OBB on warped**")
            obb_c = result.get("obb_corners")
            st.markdown(
                f"`{len(obb_c)} corners detected`" if obb_c
                else "_no OBB detection on warped_"
            )

            st.markdown("**Sharpness**")
            st.progress(min(r_sharp / 5000, 1.0), text=f"{r_sharp:.1f}")

        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════ #
#  AUTO-REFRESH                                                              #
# ══════════════════════════════════════════════════════════════════════════ #
if st.session_state["running"]:
    time.sleep(0.4)
    st.rerun()