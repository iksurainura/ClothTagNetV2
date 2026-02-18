import torch
import numpy as np
from collections import deque
from datetime import datetime
import threading
import queue
import cv2
import easyocr
import time

# ── Tunable constants ──────────────────────────────────────────────────────
STABLE_FRAMES_NEEDED = 5      # consecutive clean SEG frames before capture starts
STABILITY_TOLERANCE  = 15     # max pixel shift in mask centroid still considered stable
MIN_AREA_RATIO       = 0.02   # mask must cover at least 2% of frame
MAX_AREA_RATIO       = 0.90   # mask must cover at most 90% of frame
SEG_CONF_THRESHOLD   = 0.55   # ↑ raise this if seg picks up faces/hands/background


class DefectDetectionOCR:
    """
    Architecture
    ────────────
    1. SEG model runs continuously on every frame (triggered from frontend).
    2. Gate check: tag mask must be fully inside frame + stable for N frames.
    3. Once gate passes → capture 2-second video buffer.
    4. Pick sharpest frame from buffer.
    5. Extract 4 corners from SEG mask via minAreaRect → perspective warp.
    6. Feed warped image to OBB model → then OCR.
    7. Store result dict in self.results for frontend access.
    """

    def __init__(self, seg_model, obb_model, ocr_model=None, capture_duration=2):
        self.seg_model        = seg_model
        self.obb_model        = obb_model
        self.ocr_model        = ocr_model          # reserved, OCR handled internally
        self.capture_duration = capture_duration

        self.frame_buffer   = deque(maxlen=150)    # ~30fps × 5s headroom
        self.is_capturing   = False
        self.capture_start_time = None
        self.processing_queue   = queue.Queue()

        # Results store — thread-safe, read by frontend
        self.results       = []
        self._results_lock = threading.Lock()

        # Gate stability tracking
        self._stable_history = deque(maxlen=STABLE_FRAMES_NEEDED)  # stores mask centroids

    # ══════════════════════════════════════════════════════════════════════ #
    #  SEGMENTATION                                                          #
    # ══════════════════════════════════════════════════════════════════════ #
    def run_seg(self, frame: np.ndarray):
        """
        Run seg model on frame.
        Returns (mask_uint8, corners_4x2) or (None, None) if no detection.
        mask_uint8 : H×W uint8 binary mask (0/255)
        corners    : (4,2) float32 — ordered corners for perspective warp
        """
        results = self.seg_model(frame, conf=SEG_CONF_THRESHOLD)

        if not (results and
                hasattr(results[0], 'masks') and
                results[0].masks is not None and
                len(results[0].masks.data) > 0):
            return None, None

        # Pick highest-confidence detection (most likely the actual tag)
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else None
        best_idx = int(confs.argmax()) if confs is not None and len(confs) > 0 else 0
        mask_tensor = results[0].masks.data[best_idx].cpu().numpy()
        mask = (mask_tensor * 255).astype(np.uint8)

        # Resize mask to match original frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Extract 4 corners via minAreaRect on the mask contour
        corners = self._corners_from_mask(mask)
        return mask, corners

    def _corners_from_mask(self, mask: np.ndarray):
        """
        Find the 4-point rotated bounding box of the segmentation mask,
        equivalent to what OBB gives but derived from the seg contour.
        Returns (4,2) float32 array or None.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)          # (center, (w,h), angle)
        corners = cv2.boxPoints(rect)               # (4,2) float32
        return corners.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════ #
    #  GATE CHECK                                                            #
    # ══════════════════════════════════════════════════════════════════════ #
    def check_tag_policy(self, frame: np.ndarray):
        """
        Run seg model and enforce policies:
          1. Mask exists (tag detected)
          2. All 4 mask corners are inside the frame boundary (no clipping)
          3. Mask area is within [MIN_AREA_RATIO, MAX_AREA_RATIO] of frame
          4. Mask centroid is stable across STABLE_FRAMES_NEEDED frames

        Returns (passed: bool, mask, corners)
        """
        mask, corners = self.run_seg(frame)

        if mask is None or corners is None:
            self._stable_history.clear()
            return False, None, None

        h, w = frame.shape[:2]
        frame_area = h * w

        # Policy 1 — all corners inside frame (no edge clipping)
        for x, y in corners:
            if x < 0 or y < 0 or x >= w or y >= h:
                self._stable_history.clear()
                return False, None, None

        # Policy 2 — area sanity
        mask_area   = cv2.countNonZero(mask)
        area_ratio  = mask_area / frame_area
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            self._stable_history.clear()
            return False, None, None

        # Policy 3 — stability via centroid tracking
        M        = cv2.moments(mask)
        if M["m00"] == 0:
            self._stable_history.clear()
            return False, None, None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        self._stable_history.append((cx, cy))

        if len(self._stable_history) < STABLE_FRAMES_NEEDED:
            return False, None, None

        ref_cx, ref_cy = self._stable_history[0]
        for pcx, pcy in list(self._stable_history)[1:]:
            if abs(pcx - ref_cx) > STABILITY_TOLERANCE or \
               abs(pcy - ref_cy) > STABILITY_TOLERANCE:
                return False, None, None

        # All policies passed
        return True, mask, corners

    # ══════════════════════════════════════════════════════════════════════ #
    #  CAPTURE CONTROL                                                       #
    # ══════════════════════════════════════════════════════════════════════ #
    def start_capture(self):
        self.is_capturing       = True
        self.capture_start_time = datetime.now()
        self.frame_buffer.clear()
        self._stable_history.clear()
        print(f"[CAPTURE] Started at {self.capture_start_time.strftime('%H:%M:%S')}")

    def should_stop_capture(self) -> bool:
        if not self.is_capturing:
            return False
        elapsed = (datetime.now() - self.capture_start_time).total_seconds()
        return elapsed >= self.capture_duration

    # ══════════════════════════════════════════════════════════════════════ #
    #  SHARPNESS                                                             #
    # ══════════════════════════════════════════════════════════════════════ #
    def calc_sharpness(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
        """Laplacian + Sobel combined sharpness score."""
        if mask is not None:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            masked = frame
        gray      = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        return lap_var * 0.7 + sobel_var * 0.3

    # Keep old name as alias so nothing breaks
    def Sharpest_frame(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
        return self.calc_sharpness(frame, mask)

    def get_sharpest_frame(self, buffer=None):
        src = buffer if buffer is not None else self.frame_buffer
        if not src:
            return None
        return max(src, key=lambda x: x['sharpness'])

    # ══════════════════════════════════════════════════════════════════════ #
    #  PERSPECTIVE TRANSFORM                                                 #
    # ══════════════════════════════════════════════════════════════════════ #
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort 4 points → [top-left, top-right, bottom-right, bottom-left]."""
        pts  = corners.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        s        = pts.sum(axis=1)
        rect[0]  = pts[np.argmin(s)]    # top-left
        rect[2]  = pts[np.argmax(s)]    # bottom-right
        diff     = np.diff(pts, axis=1)
        rect[1]  = pts[np.argmin(diff)] # top-right
        rect[3]  = pts[np.argmax(diff)] # bottom-left
        return rect

    def apply_perspective_transform(self, frame: np.ndarray,
                                    corners: np.ndarray) -> np.ndarray:
        """Warp tag region to a flat front-facing rectangle."""
        rect        = self._order_corners(corners)
        tl, tr, br, bl = rect
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        dst = np.array([[0, 0], [out_w-1, 0],
                        [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (out_w, out_h))

    # ══════════════════════════════════════════════════════════════════════ #
    #  OCR                                                                   #
    # ══════════════════════════════════════════════════════════════════════ #
    def extract_text(self, image: np.ndarray) -> str:
        """Run EasyOCR on a numpy image and return lowercase paragraph."""
        start  = time.time()
        reader = easyocr.Reader(['en'], gpu=True)
        texts  = reader.readtext(image, detail=0)
        elapsed = time.time() - start
        result  = " ".join(texts).lower()
        print(f"[OCR] {elapsed:.2f}s → '{result}'")
        return result

    # ══════════════════════════════════════════════════════════════════════ #
    #  MAIN PROCESSING  (called once after capture ends)                    #
    # ══════════════════════════════════════════════════════════════════════ #
    def process_sharpest_frame(self, best: dict) -> dict:
        """
        Takes the sharpest buffered frame dict:
          {'frame': np.ndarray, 'mask': np.ndarray, 'corners': np.ndarray, 'sharpness': float}

        Steps:
          1. Perspective warp using SEG corners
          2. OBB model on warped image
          3. OCR on warped image
          4. Save to self.results
        """
        frame   = best['frame']
        mask    = best['mask']
        corners = best['corners']

        print("[PROCESS] Perspective warp → OBB → OCR")

        # Step 1 — perspective warp
        if corners is not None:
            warped = self.apply_perspective_transform(frame, corners)
        else:
            # Fallback: crop bounding rect of mask
            if mask is not None:
                x, y, w, h = cv2.boundingRect(mask)
                warped = frame[y:y+h, x:x+w]
            else:
                warped = frame

        # Step 2 — OBB on warped image
        obb_results = self.obb_model(warped)
        obb_corners = None
        if (obb_results and
                hasattr(obb_results[0], 'obb') and
                obb_results[0].obb is not None and
                len(obb_results[0].obb.xyxyxyxy) > 0):
            obb_corners = obb_results[0].obb.xyxyxyxy[0].cpu().numpy().tolist()

        # Step 3 — OCR on warped image
        ocr_text = self.extract_text(warped)

        result = {
            'id'          : len(self.results),
            'timestamp'   : datetime.now().isoformat(),
            'frame'       : frame,           # original sharpest frame
            'mask'        : mask,            # seg mask
            'warped_frame': warped,          # perspective-corrected tag
            'seg_corners' : corners.tolist() if corners is not None else None,
            'obb_corners' : obb_corners,     # OBB on warped image
            'ocr_result'  : ocr_text,
            'sharpness'   : best['sharpness'],
        }

        with self._results_lock:
            self.results.append(result)

        print(f"[RESULT] #{result['id']} saved | sharpness={result['sharpness']:.1f} | ocr='{ocr_text}'")
        return result

    # ══════════════════════════════════════════════════════════════════════ #
    #  FRONTEND ACCESS                                                       #
    # ══════════════════════════════════════════════════════════════════════ #
    def get_all_results(self) -> list:
        with self._results_lock:
            return list(self.results)

    def get_latest_result(self) -> dict:
        with self._results_lock:
            return self.results[-1] if self.results else None

    def clear_results(self):
        with self._results_lock:
            self.results.clear()