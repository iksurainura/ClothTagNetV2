import streamlit as st
import cv2
from ultralytics import YOLO
import time
import numpy as np

@st.cache_resource
def load_models():
    seg=YOLO("")
    obb=YOLO("")
    return seg , obb

seg_model,cls_model=load_models()

st.title("defect detection System")

def get_sharpness(frame,mask=None):
    masked=cv2.bitwise_and(frame,frame,mask=mask)
    gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    lap_var=cv2.Laplacian(gray,cv2.CV_64F).var()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
    sharpest=lap_var * 0.7 + sobel_var * 0.3
    return sharpest

def draw_segmentation(self,frame,results):
    if results.masks is None:
        return frame
    annotated_frame = results.plot()
    return annotated_frame


