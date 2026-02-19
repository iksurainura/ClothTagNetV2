# defect_detector.py
"""
Core defect detection logic with state machine for conveyor belt inspection
"""

import torch
import numpy as np
from collections import deque
from datetime import datetime
import threading
import cv2
import easyocr
import time
import warnings
from enum import Enum, auto

class SystemState(Enum):
    IDLE = auto()           # Seg running, watching for triggers
    CAPTURING = auto()      # 2s buffer filling
    PROCESSING = auto()     # Finding sharpest + OBB + OCR

# ── Tunable constants ──────────────────────────────────────────────────────
STABLE_FRAMES_NEEDED = 5
STABILITY_TOLERANCE  = 15       # pixels
MIN_AREA_RATIO       = 0.015    # slightly lowered — helps with small/distant tags
MAX_AREA_RATIO       = 0.92
SEG_CONF_THRESHOLD   = 0.57     # small increase — reduces false positives
CAPTURE_DURATION     = 2.0      # seconds

# Global OCR reader for sharing across instances
GLOBAL_OCR_READER = None


class DefectDetectionOCR:
    def __init__(self, seg_model, obb_model, capture_duration=CAPTURE_DURATION):
        self.seg_model = seg_model
        self.obb_model = obb_model
        self.capture_duration = capture_duration

        self.frame_buffer = deque(maxlen=180)   # ~6s @ 30 fps
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock()
        self.capture_start_time = None

        self.results = []
        self._results_lock = threading.Lock()

        self._stable_history = deque(maxlen=STABLE_FRAMES_NEEDED)
        self._ocr_reader = None
        self.processing_thread = None
        
        # For tracking current frame info
        self.current_mask = None
        self.current_corners = None
        self.current_sharpness = 0.0

    @property
    def ocr_reader(self):
        global GLOBAL_OCR_READER
        if GLOBAL_OCR_READER is not None:
            self._ocr_reader = GLOBAL_OCR_READER
        if self._ocr_reader is None:
            print("[OCR] Initializing EasyOCR (this may take several seconds)...")
            try:
                self._ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
                GLOBAL_OCR_READER = self._ocr_reader
            except Exception as e:
                warnings.warn(f"EasyOCR initialization failed: {e}\nFalling back to CPU.")
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                GLOBAL_OCR_READER = self._ocr_reader
        return self._ocr_reader

    def run_seg(self, frame: np.ndarray):
        try:
            results = self.seg_model(frame, conf=SEG_CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"[SEG] Model inference failed: {e}")
            return None, None

        if not results or len(results) == 0 or results[0].masks is None:
            return None, None

        masks = results[0].masks.data
        if len(masks) == 0:
            return None, None

        # Select best mask by confidence × area
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            areas = (results[0].boxes.xywh[:,2] * results[0].boxes.xywh[:,3]).cpu().numpy()
            scores = confs * areas
            best_idx = scores.argmax()
        else:
            best_idx = 0

        mask_tensor = masks[best_idx].cpu().numpy()
        mask = (mask_tensor > 0.5).astype(np.uint8) * 255

        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        corners = self._corners_from_mask(mask)
        return mask, corners

    def _corners_from_mask(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 150:
            return None

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    def check_tag_policy(self, frame: np.ndarray):
        """Check if tag meets criteria for capture - only in IDLE state"""
        mask, corners = self.run_seg(frame)
        
        # Store for visualization regardless of state
        self.current_mask = mask
        self.current_corners = corners
        
        if mask is None or corners is None:
            self._stable_history.clear()
            return False, None, None

        h, w = frame.shape[:2]
        frame_area = h * w

        # Corners in bounds?
        if not all(0 <= x < w and 0 <= y < h for x, y in corners):
            self._stable_history.clear()
            return False, None, None

        # Area ratio check
        mask_area = cv2.countNonZero(mask)
        area_ratio = mask_area / frame_area
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            self._stable_history.clear()
            return False, None, None

        # Centroid stability
        M = cv2.moments(mask)
        if M["m00"] < 1e-6:
            self._stable_history.clear()
            return False, None, None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        self._stable_history.append((cx, cy))

        if len(self._stable_history) < STABLE_FRAMES_NEEDED:
            return False, None, None

        ref_cx, ref_cy = self._stable_history[0]
        stable = all(
            max(abs(c[0] - ref_cx), abs(c[1] - ref_cy)) <= STABILITY_TOLERANCE
            for c in list(self._stable_history)[1:]
        )

        return stable, mask, corners

    def transition_to_capturing(self):
        with self.state_lock:
            self.state = SystemState.CAPTURING
        self.capture_start_time = datetime.now()
        self.frame_buffer.clear()
        self._stable_history.clear()
        print(f"[CAPTURE START] {self.capture_start_time.strftime('%H:%M:%S.%f')[:-3]}")

    def should_stop_capture(self) -> bool:
        if self.state != SystemState.CAPTURING:
            return False
        elapsed = (datetime.now() - self.capture_start_time).total_seconds()
        return elapsed >= self.capture_duration

    def calc_sharpness(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
        if mask is not None and cv2.countNonZero(mask) > 100:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            masked = frame

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
        tenengrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0).var() + cv2.Sobel(gray, cv2.CV_64F, 0, 1).var()
        return 0.5 * lap + 0.3 * sobel + 0.2 * tenengrad

    def get_sharpest_frame(self, buffer=None):
        buf = buffer if buffer is not None else self.frame_buffer
        if not buf:
            return None
        return max(buf, key=lambda x: x.get('sharpness', 0))

    def apply_perspective_transform(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if corners is None:
            return frame

        rect = self._order_corners(corners)
        tl, tr, br, bl = rect

        width_a = np.linalg.norm(tr - tl)
        width_b = np.linalg.norm(br - bl)
        height_a = np.linalg.norm(bl - tl)
        height_b = np.linalg.norm(br - tr)

        out_w = max(int(width_a), int(width_b))
        out_h = max(int(height_a), int(height_b))

        if out_w < 16 or out_h < 16:
            return frame

        dst = np.array([
            [0, 0],
            [out_w-1, 0],
            [out_w-1, out_h-1],
            [0, out_h-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        return warped

    def _order_corners(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left
        return rect

    def extract_text(self, image: np.ndarray) -> str:
        if image.size == 0:
            return ""

        start = time.perf_counter()
        try:
            texts = self.ocr_reader.readtext(image, detail=0, paragraph=True)
            result = " ".join(t.strip() for t in texts if t.strip()).lower()
        except Exception as e:
            print(f"[OCR ERROR] {e}")
            result = ""

        elapsed = time.perf_counter() - start
        if result:
            print(f"[OCR] {elapsed:.2f}s → {result!r}")
        return result

    def process_sharpest_frame(self, best: dict) -> dict:
        frame = best['frame']
        mask = best.get('mask')
        corners = best.get('corners')

        print("[PROCESS] warp → obb → ocr")

        warped = self.apply_perspective_transform(frame, corners)

        # OBB detection
        obb_corners = None
        try:
            obb_results = self.obb_model(warped, verbose=False)
            if obb_results and obb_results[0].obb is not None:
                first_obb = obb_results[0].obb
                if len(first_obb.xyxyxyxy) > 0:
                    obb_corners = first_obb.xyxyxyxy[0].cpu().numpy().tolist()
        except Exception as e:
            print(f"[OBB] inference failed: {e}")

        # OCR
        ocr_text = self.extract_text(warped)

        result = {
            'id': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'frame': frame,
            'mask': mask,
            'warped_frame': warped,
            'seg_corners': corners.tolist() if corners is not None else None,
            'obb_corners': obb_corners,
            'ocr_result': ocr_text,
            'sharpness': best.get('sharpness', 0.0),
        }

        with self._results_lock:
            self.results.append(result)

        print(f"[RESULT #{result['id']}] sharpness={result['sharpness']:.1f} | ocr='{ocr_text}'")
        return result

    def transition_to_processing(self):
        """Hand off to background thread"""
        with self.state_lock:
            self.state = SystemState.PROCESSING
        
        buffer_snapshot = list(self.frame_buffer)
        self.frame_buffer.clear()
        
        self.processing_thread = threading.Thread(
            target=self._process_capture_batch,
            args=(buffer_snapshot,),
            daemon=True
        )
        self.processing_thread.start()

    def _process_capture_batch(self, buffer: list):
        """Background processing"""
        try:
            best = self.get_sharpest_frame(buffer)
            if best is None:
                print("[ERROR] Empty buffer")
                return
            
            self.process_sharpest_frame(best)
            time.sleep(0.3)  # Debounce
            
        finally:
            with self.state_lock:
                self.state = SystemState.IDLE
                print("[STATE] Returned to IDLE")

    def update(self, frame: np.ndarray):
        """Main update method - call every frame"""
        with self.state_lock:
            current_state = self.state
        
        # Always run seg first for visualization
        mask, corners = self.run_seg(frame)
        self.current_mask = mask
        self.current_corners = corners
        
        sharpness = self.calc_sharpness(frame, mask) if mask is not None else 0
        self.current_sharpness = sharpness
        
        if current_state == SystemState.IDLE:
            stable, _, _ = self.check_tag_policy(frame)
            if stable:
                with self.state_lock:
                    if self.state == SystemState.IDLE:
                        self.transition_to_capturing()
            
            return {
                'seg_mask': mask,
                'seg_corners': corners,
                'status': 'IDLE',
                'sharpness': sharpness,
                'buffer_count': 0
            }
        
        elif current_state == SystemState.CAPTURING:
            self.frame_buffer.append({
                'frame': frame.copy(),
                'mask': mask.copy() if mask is not None else None,
                'corners': corners.copy() if corners is not None else None,
                'sharpness': sharpness,
                'timestamp': datetime.now()
            })
            
            if self.should_stop_capture():
                self.transition_to_processing()
            
            return {
                'seg_mask': mask,
                'seg_corners': corners,
                'status': 'CAPTURING',
                'sharpness': sharpness,
                'buffer_count': len(self.frame_buffer)
            }
        
        elif current_state == SystemState.PROCESSING:
            return {
                'seg_mask': mask,
                'seg_corners': corners,
                'status': 'PROCESSING',
                'sharpness': sharpness,
                'buffer_count': 0
            }

    def get_all_results(self):
        with self._results_lock:
            return list(self.results)

    def get_latest_result(self):
        with self._results_lock:
            return self.results[-1] if self.results else None

    def clear_results(self):
        with self._results_lock:
            self.results.clear()