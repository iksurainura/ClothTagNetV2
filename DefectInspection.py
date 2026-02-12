from ultralytics import YOLO
import cv2
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Dict
import time


@dataclass
class InspectionResult:
    label: str
    confidence: float
    extracted_image: np.ndarray


@dataclass
class SystemConfig:
    buffer_time: float = 2.0
    seg_conf: float = 0.7


class DefectInspection:
    def __init__(self, config: SystemConfig = None, model_path: str = ""):
        self.config = config or SystemConfig()
        self.model_path = model_path
        self.cap = None
        self.start = False
        self.frame_buffer: List[Dict] = []
        self.latest_result: Optional[InspectionResult] = None
    
    def load_models(self):
        if not self.model_path:
            raise ValueError("Model path not provided")
        self.obb_model = YOLO(self.model_path)

    def start_camera(self, source: int = 0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def Sharpest_frame(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
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

    def run(self):
        if not hasattr(self, 'obb_model'):
            raise RuntimeError("Model not loaded. Call load_models() first.")
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not started. Call start_camera() first.")
        self.start = True

        while self.start:
            # Start fresh cycle - clear buffer
            self.frame_buffer = []
            cycle_start_time = time.time()
            
            # Capture frames for exactly buffer_time seconds
            while self.start and (time.time() - cycle_start_time) < self.config.buffer_time:
                ret, frame = self.cap.read()
                if not ret:
                    self.start = False
                    break
                
                # Calculate sharpness immediately and store
                sharpness = self.Sharpest_frame(frame)
                self.frame_buffer.append({
                    'frame': frame,
                    'sharpness': sharpness,
                })
            
            if not self.start or len(self.frame_buffer) == 0:
                break
            
            # Get sharpest frame using max with key
            best = max(self.frame_buffer, key=lambda x: x['sharpness'])
            sharpest_frame = best['frame']
            
            # Run model prediction on sharpest frame
            if hasattr(self, 'obb_model'):
                results = self.obb_model.predict(
                    sharpest_frame,
                    conf=self.config.seg_conf
                )
                
                # Process results - check for both OBB and regular boxes
                has_detections = False
                boxes_data = None
                
                if len(results) > 0:
                    # Try OBB first (Oriented Bounding Boxes)
                    if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
                        boxes_data = results[0].obb
                        has_detections = True
                    # Try regular boxes
                    elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes_data = results[0].boxes
                        has_detections = True
                
                if has_detections and boxes_data is not None:
                    best_idx = boxes_data.conf.argmax().item()
                    
                    self.latest_result = InspectionResult(
                        label=results[0].names[int(boxes_data.cls[best_idx])],
                        confidence=float(boxes_data.conf[best_idx]),
                        extracted_image=sharpest_frame.copy()
                    )
                    
                    print(f"Detected: {self.latest_result.label} "
                          f"({self.latest_result.confidence:.2f})")
                else:
                    self.latest_result = InspectionResult(
                        label="no_detection",
                        confidence=0.0,
                        extracted_image=sharpest_frame.copy()
                    )
                    print("No defects detected")

    def stop(self):
        self.start = False
        if self.cap:
            self.cap.release()
            self.cap = None