import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np

@st.cache_resource
def load_models():
    return YOLO(""),YOLO()

seg_model,cls_model=load_models()

st.title("defect detection System")

col1, col2 = st.columns([2, 1])


def run_inspection():
    cap = cv2.VideoCapture(0)
    is_bursting = False
    burst_timer = 0
    # The 'Winner' container for our collection
    winner = {'img': None, 'score': -1}

    while True:
        ret, frame = cap.read()
        if not ret: break
        now = time.time()

        # Step 1: Segmentation (The Trigger)
        results = seg_model(frame, conf=0.7, verbose=False)[0]
        detected = len(results.masks) > 0 if results.masks else False

        # Start 2-second collection if object appears
        if detected and not is_bursting:
            is_bursting = True
            burst_timer = now
            winner = {'img': None, 'score': -1} # Reset collection

        # Step 2: Collection Logic (The 2-second window)
        if is_bursting:
            if (now - burst_timer) <= 2.0:
                # Calculate sharpness of CURRENT frame in collection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Check if this frame is the sharpest in the collection so far
                if score > winner['score']:
                    winner = {'img': frame.copy(), 'score': score}
            else:
                # 2 seconds are up! Process the winner
                if winner['img'] is not None:
                    # Final OBB Detection on the ONE sharpest frame
                    obb_res = obb_model(winner['img'], conf=0.5, verbose=False)[0]
                    
                    # Draw OBB results on the winner image
                    annotated = obb_res.plot()
                    
                    # Show ONLY this frame on the frontend with OBB annotations
                    hero_view.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                    caption=f"SHARPEST FRAME | Score: {winner['score']:.1f}")
                    
                    # Display OBB results in toast
                    if len(obb_res.obb) > 0:
                        classes = [obb_res.names[int(cls)] for cls in obb_res.obb.cls]
                        confs = obb_res.obb.conf.tolist()
                        result_text = " | ".join([f"{c}: {conf:.2f}" for c, conf in zip(classes, confs)])
                        st.toast(f"OBB Detected: {result_text}")
                    else:
                        st.toast("No oriented objects detected")
                
                is_bursting = False # Reset and wait for next trigger

        # Update live monitor
        live_view.image(cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB), width=500)

    cap.release()

run_inspection()