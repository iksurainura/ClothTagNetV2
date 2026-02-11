import cv2, time, numpy as np, streamlit as st
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Industrial Collection Hub")

# --- PERSISTENT STORAGE ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Stores: {'img': annotated_frame, 'score': val, 'time': str}

# --- UI LAYOUT ---
st.title("🏭 Triggered Inspection Gallery")
col_live, col_gallery = st.columns([1, 1.2])

live_view = col_live.empty()
with col_gallery:
    st.subheader("📦 Collection History (Sharpest Only)")
    gallery_container = st.container()

@st.cache_resource
def load_models():
    return YOLO("yolo11n-seg.pt"), YOLO("yolo11m-cls.pt")

seg_model, cls_model = load_models()

def run_app():
    cap = cv2.VideoCapture(0)
    is_bursting = False
    burst_start = 0
    winner = {'frame': None, 'score': -1}
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        now = time.time()

        # 1. SEGMENTATION (The Trigger)
        res = seg_model(frame, conf=0.7, verbose=False)[0]
        mask_active = len(res.masks) > 0 if res.masks else False

        # Start 2s Burst if object detected
        if mask_active and not is_bursting:
            is_bursting = True
            burst_start = now
            winner = {'frame': None, 'score': -1}

        # 2. THE 2-SECOND COLLECTION
        if is_bursting:
            if (now - burst_start) <= 2.0:
                # Score sharpness inside the mask
                m_data = res.masks.data[0].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(m_data, (frame.shape[1], frame.shape[0]))
                gray = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if score > winner['score']:
                    winner = {'frame': frame.copy(), 'score': score}
            else:
                # CYCLE END: Classify & Annotate the Sharpest
                if winner['frame'] is not None:
                    # Run classification
                    cls_res = cls_model.predict(winner['frame'], verbose=False)[0]
                    # This line creates the "output image" with the label drawn on it
                    annotated_img = cls_res.plot() 
                    
                    # Add to history
                    st.session_state.history.insert(0, {
                        'img': cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        'score': winner['score'],
                        'time': time.strftime("%H:%M:%S")
                    })
                
                is_bursting = False # Reset

        # 3. RENDER FRONTEND
        # Live Stream
        live_view.image(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB), width=450, caption="Live Monitor")
        
        # Grid Gallery (Top 6 results)
        with gallery_container:
            cols = st.columns(2)
            for idx, item in enumerate(st.session_state.history[:6]):
                with cols[idx % 2]:
                    st.image(item['img'], use_container_width=True, 
                             caption=f"🕒 {item['time']} | Sharpness: {item['score']:.1f}")

    cap.release()

run_app()