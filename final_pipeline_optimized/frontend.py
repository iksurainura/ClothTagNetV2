"""
frontendOCR.py  —  Streamlit frontend for DefectDetectionOCR
Run with:  streamlit run frontendOCR.py
DefectDetectionOCR.py must be in the same directory.

Pipeline
────────
START → SEG gate (every 3rd frame, FP16) → 2s capture → sharpest frame
      → perspective warp → OBB (FP16) → OCR → results
"""

import streamlit as st
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import final_pipeline_optimized.DefectDetection as DefectDetection

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
    .metric-label {
        font-size: 11px; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-value {
        font-size: 28px; font-weight: 700; color: #58a6ff; margin-top: 4px;
    }

    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; letter-spacing: .4px;
    }
    .badge-green  { background:#0d2a1a; color:#3fb950; border:1px solid #3fb950; }
    .badge-yellow { background:#2a2200; color:#d29922; border:1px solid #d29922; }
    .badge-blue   { background:#0d1f33; color:#58a6ff; border:1px solid #58a6ff; }
    .badge-red    { background:#2a0d0d; color:#f85149; border:1px solid #f85149; }

    .sec-hdr {
        font-size: 13px; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        border-bottom: 1px solid #21262d;
        padding-bottom: 5px; margin: 14px 0 10px;
    }

    .gate-log {
        background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 12px; font-family: monospace; font-size: 11px;
        color: #8b949e; height: 220px; overflow-y: auto;
    }

    .result-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 16px; margin-bottom: 14px;
    }
    .result-hdr {
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 10px; border-bottom: 1px solid #21262d;
        margin-bottom: 12px;
    }
    .result-id { font-size: 16px; font-weight: 700; color: #58a6ff; }
    .result-ts { font-size: 11px; color: #8b949e; }

    .ocr-box {
        background: #0d1117; border-left: 3px solid #58a6ff;
        border-radius: 0 6px 6px 0; padding: 10px 12px;
        font-family: monospace; font-size: 13px; color: #e0e0e0;
        word-break: break-all; min-height: 40px;
    }

    .gpu-chip {
        display: inline-block; background: #1a2a1a;
        border: 1px solid #3fb950; border-radius: 4px;
        padding: 2px 8px; font-size: 11px; color: #3fb950;
        font-family: monospace;
    }
    .pipe-step {
        display: inline-block; background: #21262d;
        border-radius: 4px; padding: 2px 8px;
        font-size: 11px; color: #8b949e; margin-right: 4px;
    }
    .pipe-arrow { color: #3fb950; font-weight: 700; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════ #
#  THREAD-SAFE SHARED STATE                                                  #
#  Camera thread and inference thread write here.                            #
#  Main Streamlit thread reads here on every rerun.                         #
#  Never touch st.session_state from background threads.                    #
# ══════════════════════════════════════════════════════════════════════════ #
_lock   = threading.Lock()
_shared = {
    "live_frame"  : None,       # RGB np.ndarray — raw camera frame
    "seg_overlay" : None,       # RGB np.ndarray — frame + green mask overlay
    "gate_status" : "idle",     # idle | waiting | capturing | processing
    "gate_log"    : [],
    "frame_count" : 0,
    "capture_pct" : 0.0,
    "gpu_active"  : False,      # True when CUDA available
    "inf_fps"     : 0.0,        # inference frames per second estimate
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
    seg = YOLO(seg_path)
    obb = YOLO(obb_path)
    return seg, obb


# ══════════════════════════════════════════════════════════════════════════ #
#  SEG OVERLAY  — uses YOLO's built-in .plot() renderer                     #
#  Draws masks, boxes, confidence labels cleanly in one call.               #
#  No manual overlay code needed — same approach as the reference snippet.  #
# ══════════════════════════════════════════════════════════════════════════ #
def seg_plot_to_rgb(seg_result, corners: np.ndarray = None) -> np.ndarray:
    """
    Call YOLO result.plot() to get the annotated BGR frame,
    then optionally draw the 4 warp-corner dots on top,
    then convert to RGB for Streamlit.
    """
    annotated = seg_result.plot()   # YOLO draws masks + boxes + labels
    if corners is not None:
        pts = corners.astype(np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 80), 2)
        for pt in pts:
            cv2.circle(annotated, tuple(pt), 6, (0, 255, 80), -1)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════ #
#  INFERENCE LOOP  (Thread 2 — runs seg/OBB/OCR, never blocks live feed)   #
# ══════════════════════════════════════════════════════════════════════════ #
def inference_loop(detector: DefectDetection,
                   frame_queue: queue.Queue,
                   stop_event: threading.Event):
    """
    Pulls latest frame from queue.
    During gate phase: runs seg every GATE_SKIP_FRAMES (handled inside class).
    During capture phase: runs seg every frame to build sharpest buffer.
    """
    inf_times = deque_import = __import__('collections').deque(maxlen=10)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue

        t0 = time.perf_counter()

        # ── Gate phase ────────────────────────────────────────────────────
        if not detector.is_capturing:
            _set("gate_status", "waiting")
            passed, mask, corners, seg_result = detector.check_tag_policy(frame)

            # .plot() draws masks + boxes + labels — no manual overlay needed
            if seg_result is not None:
                _set("seg_overlay", seg_plot_to_rgb(seg_result, corners))

            if passed:
                _log("✅ Gate passed — starting capture")
                detector.start_capture()
                _set("gate_status", "capturing")
            else:
                if mask is not None:
                    _log("⏳ Tag visible — waiting for stability...")

        # ── Capture phase ─────────────────────────────────────────────────
        else:
            _set("gate_status", "capturing")
            mask, corners, seg_result = detector.run_seg(frame)
            sharpness = detector.calc_sharpness(frame, mask)

            # Keep live feed annotated during capture too
            if seg_result is not None:
                _set("seg_overlay", seg_plot_to_rgb(seg_result, corners))

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
#  CAMERA LOOP  (Thread 1 — ONLY reads frames, zero inference here)        #
# ══════════════════════════════════════════════════════════════════════════ #
def camera_loop(detector: DefectDetection,
                cam_idx: int,
                stop_event: threading.Event):
    """
    Reads camera at full fps.
    Live feed updates instantly — inference never blocks it.
    CAP_PROP_BUFFERSIZE=1 ensures we always get the freshest frame.
    """
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # always freshest frame

    # maxsize=1: inference always gets latest frame, never stale ones
    frame_q = queue.Queue(maxsize=1)

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

        # Live feed: full fps, no inference blocking it
        _set("live_frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Push to inference queue — drop stale frame if inference busy
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
    cls = {"green":"badge-green","yellow":"badge-yellow",
           "blue":"badge-blue","red":"badge-red"}.get(color,"badge-blue")
    return f'<span class="badge {cls}">{text}</span>'

def status_badge():
    s = _get("gate_status")
    m = {
        "idle"      : ("⏸ Idle",       "yellow"),
        "waiting"   : ("👁 Watching",   "blue"),
        "capturing" : ("🎬 Capturing",  "green"),
        "processing": ("⚙️ Processing", "yellow"),
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

    # GPU status chip
    import torch
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "Not available"
    chip_color = "#3fb950" if gpu_ok else "#f85149"
    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'<span style="font-size:11px;color:#8b949e">GPU &nbsp;</span>'
        f'<span class="gpu-chip" style="border-color:{chip_color};color:{chip_color}">'
        f'{"✓ " + gpu_name if gpu_ok else "✗ CPU only"}</span>'
        f'</div>',
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

    # Push to module constants live
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

    # Pipeline diagram
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
        'FP16 · no_grad · imgsz fixed · buf=1 · warmup</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  BUTTON ACTIONS                                                            #
# ══════════════════════════════════════════════════════════════════════════ #
if start_btn and not st.session_state["running"]:
    try:
        seg_model, obb_model = load_models(SEG_MODEL_PATH, OBB_MODEL_PATH)

        # DefectDetectionOCR.__init__ runs warmup automatically
        detector = DefectDetection(
            seg_model=seg_model,
            obb_model=obb_model,
            capture_duration=cap_dur,
        )
        st.session_state["detector"]   = detector
        st.session_state["running"]    = True
        st.session_state["stop_event"] = threading.Event()

        _set("gate_log",    [])
        _set("frame_count", 0)
        _set("gate_status", "waiting")
        _set("live_frame",  None)
        _set("seg_overlay", None)
        _set("capture_pct", 0.0)
        _set("gpu_active",  gpu_ok)

        t = threading.Thread(
            target=camera_loop,
            args=(detector, int(cam_idx), st.session_state["stop_event"]),
            daemon=True,
        )
        t.start()
        st.success(f"✅ Started! GPU: {'ON (FP16)' if gpu_ok else "OFF (CPU)"}")
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

# ── Metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(metric("Frames", _get("frame_count")), unsafe_allow_html=True)
with m2:
    st.markdown(metric("Tags", len(all_results)), unsafe_allow_html=True)
with m3:
    avg = np.mean([r["sharpness"] for r in all_results]) if all_results else 0
    st.markdown(metric("Avg Sharp", f"{avg:.0f}"), unsafe_allow_html=True)
with m4:
    inf_fps = _get("inf_fps")
    st.markdown(metric("Inf FPS", f"{inf_fps:.1f}"), unsafe_allow_html=True)
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
    st.markdown('<div class="sec-hdr">📷 Live Feed (SEG overlay)</div>',
                unsafe_allow_html=True)
    overlay = _get("seg_overlay")
    raw     = _get("live_frame")
    display = overlay if overlay is not None else raw
    if display is not None:
        st.image(display, use_column_width=True)
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
        st.markdown(f"**Last OCR:** `{all_results[-1]['ocr_result'] or 'none'}`")

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
            # Show OBB-annotated warped frame (.plot() output) if available,
            # otherwise fall back to plain warped frame
            display_img = result.get("obb_annotated") or result.get("warped_frame")
            if display_img is not None:
                display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                caption = "Warped tag + OBB detections" \
                          if result.get("obb_annotated") is not None \
                          else "Perspective-corrected tag"
                st.image(display_rgb, caption=caption, use_column_width=True)

            tab1, tab2 = st.tabs(["Plain warped", "Original frame"])
            with tab1:
                if result.get("warped_frame") is not None:
                    st.image(cv2.cvtColor(result["warped_frame"],
                             cv2.COLOR_BGR2RGB), use_column_width=True)
            with tab2:
                if result.get("frame") is not None:
                    st.image(cv2.cvtColor(result["frame"],
                             cv2.COLOR_BGR2RGB), use_column_width=True)

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
                f"`{len(obb_c)} corners`" if obb_c
                else "_no detection on warped_"
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