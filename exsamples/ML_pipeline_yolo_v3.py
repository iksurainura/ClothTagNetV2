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
        
        # TIMING CONTROL
        self.segmentation_active = True
        self.cycle_start_time = time.time()
        self.in_cooldown = False
        
        # Results
        self.last_result = None
        self.display_until = 0
        
        # Colors for different classes (random colors)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
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
    
    def draw_segmentation(self, frame, results):
        """Draw segmentation masks and labels on frame."""
        if results.masks is None:
            return frame
        # Get the annotated frame from Ultralytics (includes masks and boxes)
        annotated_frame = results.plot()
        
        return annotated_frame
    
    def extract_and_classify(self, sharpest_data):
        """Extract object and run classification."""
        frame = sharpest_data['frame']
        mask = sharpest_data['mask']
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < self.config.MIN_OBJECT_AREA:
            return None, None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        width, height = int(rect[1][0]), int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        extracted = cv2.warpPerspective(frame, matrix, (width, height))
        
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
            if elapsed >= self.config.SEGMENTATION_ACTIVE_TIME:
                print(f"[SLEEP] Active period ended. Sleeping for {self.config.SEGMENTATION_SLEEP_TIME}s...")
                self.segmentation_active = False
                self.cycle_start_time = current_time
                self.in_cooldown = True
                self.active_collection = False
                self.frame_buffer = []
        else:
            if elapsed >= self.config.SEGMENTATION_SLEEP_TIME:
                print(f"[ACTIVE] Sleep ended. Resuming detection for {self.config.SEGMENTATION_ACTIVE_TIME}s...")
                self.segmentation_active = True
                self.cycle_start_time = current_time
                self.in_cooldown = False
    
    def run(self):
        """Main loop with timing control and segmented output."""
        print(f"[START] Running: {self.config.SEGMENTATION_ACTIVE_TIME}s ON / {self.config.SEGMENTATION_SLEEP_TIME}s OFF")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.update_timing_cycle()
            
            # SLEEP MODE: Show raw feed only
            if not self.segmentation_active:
                sleep_frame = frame.copy()
                cv2.putText(sleep_frame, "SLEEP MODE", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(sleep_frame, "No segmentation", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Inspection", sleep_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # ACTIVE MODE: Run segmentation and show annotated output
            try:
                results = self.seg_model(frame, conf=self.config.SEG_CONF, verbose=False)[0]
            except:
                continue
            
            # GET SEGMENTED FRAME with masks and boxes drawn
            output_frame = self.draw_segmentation(frame, results)
            
            has_mask = results.masks is not None and len(results.masks) > 0
            
            # Handle detection and buffering logic
            if has_mask and not self.in_cooldown:
                if not self.active_collection:
                    self.active_collection = True
                    self.collection_start = time.time()
                    self.frame_buffer = []
                    print("[BUFFER] Started collecting frames...")
                
                # Collect frames during buffer window
                if time.time() - self.collection_start <= self.config.BUFFER_TIME:
                    # Use the first mask for sharpness calculation
                    mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    score = self.get_sharpness(frame, mask)
                    self.frame_buffer.append({
                        'sharpness': score,
                        'frame': frame.copy(),
                        'mask': mask
                    })
                    
                    # Draw collecting indicator on segmented frame
                    cv2.putText(output_frame, "COLLECTING FRAMES...", (10, output_frame.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    # Process best frame
                    if self.frame_buffer:
                        best = max(self.frame_buffer, key=lambda x: x['sharpness'])
                        
                        if best['sharpness'] >= self.config.MIN_SHARPNESS:
                            result, box = self.extract_and_classify(best)
                            
                            if result:
                                print(f"[RESULT] {result['label'].upper()} ({result['conf']:.1%}) [sharp: {result['sharp']:.1f}]")
                                self.last_result = result
                                self.display_until = time.time() + 3.0
                                
                                # Draw classification result on the segmented frame
                                label_text = f"DEFECT: {result['label']} ({result['conf']:.1%})"
                                cv2.putText(output_frame, label_text, (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            print(f"[REJECT] Too blurry: {best['sharpness']:.1f}")
                    
                    self.active_collection = False
                    self.frame_buffer = []
            
            # Draw status overlays on segmented frame
            status = "SCANNING" if not self.active_collection else "COLLECTING"
            color = (0, 255, 0) if not self.active_collection else (0, 165, 255)
            cv2.putText(output_frame, f"STATUS: {status}", (10, output_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show last classification result
            if self.last_result and time.time() < self.display_until:
                text = f"LAST: {self.last_result['label']} ({self.last_result['conf']:.1%})"
                cv2.putText(output_frame, text, (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show active timer
            remaining = self.config.SEGMENTATION_ACTIVE_TIME - (time.time() - self.cycle_start_time)
            cv2.putText(output_frame, f"ACTIVE: {remaining:.1f}s", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show the SEGMENTED output (not raw frame)
            cv2.imshow("Inspection", output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


# Create and run
app = DefectInspectionSystem()
app.run()