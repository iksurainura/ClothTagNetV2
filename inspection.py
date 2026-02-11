import cv2
import time
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any,List
import threading
from queue import Queue

@dataclass
class InspectionResult:
    label: str
    confidence: float
    sharpness: float
    box: np.ndarray
    extracted_image: np.ndarray

@dataclass
class SystemConfig:
    buffer_time: float = 2.0
    seg_conf: float = 0.7

class DefectInspectionCore:
    def __init__(self, start=False, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.cap = None
        self.start = start
        self.is_collecting = False      # Track if we're in the 2-second collection phase
        self.collection_start = 0
        self.frame_buffer: List[Dict] = []
        self.latest_result: Optional[InspectionResult] = None
        
        # Callback for when inspection completes (optional)
        self.on_inspection_complete: Optional[Callable[[InspectionResult], None]] = None
        
    def load_models(self):
        self.seg_model = YOLO("")
        self.obb_model = YOLO("")

    def start_camera(self,source: int = 0):
        """Initialize camera."""
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    def get_sharpness(self, frame:np.array,mask:np.ndarray=None)->float:
        """Calculate sharpness score"""
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        sharpness=lap_var * 0.7 + sobel_var * 0.3
        return sharpness
    
    def extract_and_classify(self, frame: np.ndarray, mask: np.ndarray) -> Optional[InspectionResult]:
        """Extract object and classify with OBB model."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.config.min_object_area:
            return None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        width, height = int(rect[1][0]), int(rect[1][1])
        if width == 0 or height == 0:
            return None
            
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        extracted = cv2.warpPerspective(frame, matrix, (width, height))
        
        if extracted.size == 0:
            return None
        
        # Resize for classification
        resized = cv2.resize(extracted, (224, 224))
        results = self.obb_model.predict(resized, verbose=False)[0]
        
        if hasattr(results, 'probs') and results.probs is not None:
            conf = results.probs.top1conf.item()
            if conf >= self.config.min_confidence:
                label = results.names[results.probs.top1]
                return InspectionResult(
                    label=label,
                    confidence=conf,
                    sharpness=self.get_sharpness(frame, mask),
                    box=box,
                    extracted_image=extracted
                )
        return None
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        current_time=time.time()
        while self.start:
            try:
                result=self.seg_model(frame,conf=self.config.seg_conf,verbose=False)[0]
            except Exception as e:
                return 
            has_mask=result.mask is not None and len(result.mask)>0
            if has_mask:
                self.collection_start=current_time
                self.frame_buffer=[]
            while current_time - self.collection_start <= self.config.buffer_time:
                mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                score=self.get_sharpness(frame,mask)
                self.frame_buffer.append({
                    'sharpness': score,
                    'frame': frame.copy(),
                    'mask': mask
                })
            best = max(self.frame_buffer, key=lambda x: x['sharpness'])
            result=self.obb_model(best,)

    def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        