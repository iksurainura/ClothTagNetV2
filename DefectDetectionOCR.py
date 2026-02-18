import torch
import numpy as np
from collections import deque # as like list but fast than list 
from datetime import datetime,timedelta
import threading
import queue
import cv2


class DefectDetectionOCR:
    def __init__(self,seg_model,obb_model,ocr_model,capture_duration=5):
        self.seg_model =seg_model
        self.ocr_model=ocr_model
        self.obb_model=obb_model
        self.capture_duration=capture_duration
        self.frame_buffer = deque(maxlen=150)  # Assuming ~30fps for 5 seconds
        self.is_capturing = False
        self.capture_start_time = None
        self.detection_active = False
        self.processing_queue = queue.Queue()

    def Sharpest_frame(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
        """Calculate sharpness score for a frame"""
        if mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            masked = frame
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        sharpness = lap_var * 0.7 + sobel_var * 0.3
        return sharpness

    def start_capture(self):
        self.is_capturing=True
        self.capture_start_time=datetime.now()
        self.frame_buffer.clear()
        print(f"Started capturing frames at {self.capture_start_time}")

    def should_stop_capture(self):
        if not self.is_capturing:
            return False
        elapsed =(datetime.now() - self.capture_start_time).total_seconds()
        return elapsed >= self.capture_duration
    
    def get_sharpest_frame(self,frames_with_masks):
        if not frames_with_masks:
            return None,None,0
        sharpest_frame = max(self.frame_buffer, key=lambda x: x['sharpness'])
        