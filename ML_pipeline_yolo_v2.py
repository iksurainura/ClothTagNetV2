import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
import threading

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Segmentation timing control
    SEGMENTATION_ACTIVE_TIME = 5.0      # Run detection for 5 seconds
    SEGMENTATION_SLEEP_TIME = 3.0       # Then sleep for 3 seconds
    
    # Quality thresholds
    BUFFER_TIME = 2.0                   # Collect frames for 2s when object detected
    MIN_SHARPNESS = 100.0
    MIN_CONFIDENCE = 0.7
    MIN_OBJECT_AREA = 1000
    
    # Model settings
    SEG_CONF = 0.7


# ==========================================
# MAIN APPLICATION
# ==========================================
class DefectInspectionSystem:
    def __init__(self):
        self.config = Config()
        
        print("Loading models...")
        self.seg_model = YOLO("yolo11n-seg.pt")
        self.cls_model = YOLO("yolo11m-cls.pt")
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # State management
        self.frame_buffer = []
        self.active_collection = False
        self.collection_start = 0
        
        # TIMING CONTROL VARIABLES
        self.segmentation_active = True         # Start in active mode
        self.cycle_start_time = time.time()     # Track when current cycle started
        self.in_cooldown = False                # Prevent processing during sleep
        
        # Results
        self.last_result = None
        self.display_until = 0
    
    def get_sharpness(self, frame, mask=None):
        """Calculate sharpness score."""
        if mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            gray = gray[mask > 0]
            if len(gray) == 0:
                return 0.0
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        
        return lap_var * 0.7 + sobel_var * 0.3
    
    def extract_and_classify(self, sharpest_data):
        """Extract object and run classification."""
        frame = sharpest_data['frame']
        mask = sharpest_data['mask']
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.config.MIN_OBJECT_AREA:
            return None, None
        
        # Get oriented bounding box
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Perspective transform
        width, height = int(rect[1][0]), int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        extracted = cv2.warpPerspective(frame, matrix, (width, height))
        
        # Classify
        if extracted.size == 0:
            return None, box
        
        resized = cv2.resize(extracted, (224, 224))
        results = self.cls_model.predict(resized, verbose=False)[0]
        
        if hasattr(results, 'probs'):
            conf = results.probs.top1conf.item()
            if conf >= self.config.MIN_CONFIDENCE:
                label = results.names[results.probs.top1]
                return {'label': label, 'conf': conf, 'sharp': sharpest_data['sharpness']}, box
        
        return None, box
    
    def update_timing_cycle(self):
        """Handle segmentation active/sleep timing."""
        current_time = time.time()
        elapsed = current_time - self.cycle_start_time
        
        if self.segmentation_active:
            # Check if active period expired
            if elapsed >= self.config.SEGMENTATION_ACTIVE_TIME:
                print(f"[SLEEP] Active period ended. Sleeping for {self.config.SEGMENTATION_SLEEP_TIME}s...")
                self.segmentation_active = False
                self.cycle_start_time = current_time
                self.in_cooldown = True
                self.active_collection = False  # Cancel any active collection
                self.frame_buffer = []
        else:
            # Check if sleep period expired
            if elapsed >= self.config.SEGMENTATION_SLEEP_TIME:
                print(f"[ACTIVE] Sleep ended. Resuming detection for {self.config.SEGMENTATION_ACTIVE_TIME}s...")
                self.segmentation_active = True
                self.cycle_start_time = current_time
                self.in_cooldown = False
    
    def run(self):
        """Main loop with timing control."""
        print(f"[START] Running: {self.config.SEGMENTATION_ACTIVE_TIME}s ON / {self.config.SEGMENTATION_SLEEP_TIME}s OFF")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Update timing cycle (active/sleep states)
            self.update_timing_cycle()
            
            # During sleep period: just show feed, skip all processing
            if not self.segmentation_active:
                cv2.putText(frame, "SLEEP MODE", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Inspection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # ACTIVE PERIOD: Run segmentation
            try:
                results = self.seg_model(frame, conf=self.config.SEG_CONF, verbose=False)[0]
            except:
                continue
            
            has_mask = results.masks is not None and len(results.masks) > 0
            
            # Handle detection logic
            if has_mask and not self.in_cooldown:
                if not self.active_collection:
                    # Start new collection
                    self.active_collection = True
                    self.collection_start = time.time()
                    self.frame_buffer = []
                    print("[BUFFER] Started collecting frames...")
                
                # Collect frames
                if time.time() - self.collection_start <= self.config.BUFFER_TIME:
                    mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    score = self.get_sharpness(frame, mask)
                    self.frame_buffer.append({
                        'sharpness': score,
                        'frame': frame.copy(),
                        'mask': mask
                    })
                else:
                    # Buffer full - process best frame
                    if self.frame_buffer:
                        best = max(self.frame_buffer, key=lambda x: x['sharpness'])
                        
                        if best['sharpness'] >= self.config.MIN_SHARPNESS:
                            result, box = self.extract_and_classify(best)
                            
                            if result:
                                print(f"[RESULT] {result['label'].upper()} ({result['conf']:.1%}) [sharpness: {result['sharp']:.1f}]")
                                self.last_result = result
                                self.display_until = time.time() + 2.0
                                
                                # Draw on frame
                                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                                cv2.putText(frame, f"{result['label']} ({result['conf']:.1%})", 
                                           (box[0][0], box[0][1]-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            print(f"[REJECT] Too blurry: {best['sharpness']:.1f}")
                    
                    # Reset for next detection
                    self.active_collection = False
                    self.frame_buffer = []
            
            # Draw status overlay
            status = "COLLECTING" if self.active_collection else "SCANNING"
            color = (0, 165, 255) if self.active_collection else (0, 255, 0)
            cv2.putText(frame, status, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw last result if valid
            if self.last_result and time.time() < self.display_until:
                text = f"Last: {self.last_result['label']} ({self.last_result['conf']:.1%})"
                cv2.putText(frame, text, (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show time remaining in active period
            remaining = self.config.SEGMENTATION_ACTIVE_TIME - (time.time() - self.cycle_start_time)
            cv2.putText(frame, f"Active: {remaining:.1f}s", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Inspection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


# Create and run immediately
app = DefectInspectionSystem()
app.run()