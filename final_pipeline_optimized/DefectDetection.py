import torch
import numpy as np
from collections import deque
from datetime import datetime
import threading
import queue
import cv2
import easyocr
import time

# ══════════════════════════════════════════════════════════════════════════ #
#  TUNABLE CONSTANTS                                                         #
# ══════════════════════════════════════════════════════════════════════════ #
STABLE_FRAMES_NEEDED = 2      # consecutive clean SEG frames before capture
STABILITY_TOLERANCE  = 15     # max centroid pixel shift still considered stable
MIN_AREA_RATIO       = 0.02   # tag mask must cover at least 2% of frame
MAX_AREA_RATIO       = 0.90   # tag mask must cover at most 90% of frame
SEG_CONF_THRESHOLD   = 0.55   # raise if seg picks up background/faces
SEG_IMGSZ            = 640    # fixed input size — skips per-frame resize logic
GATE_SKIP_FRAMES     = 3      # run seg every Nth frame during gate phase


class DefectDetection:
    """
    GPU-Optimised Pipeline
    ──────────────────────
    1.  SEG model watches every Nth frame (gate phase, FP16, verbose off)
    2.  Gate: tag fully inside frame + stable centroid for N frames
    3.  Once gate passes → capture buffer for capture_duration seconds
    4.  Sharpest frame selected from buffer
    5.  Perspective warp from SEG mask corners (minAreaRect)
    6.  OBB model on warped image (FP16, verbose off)
    7.  OCR on warped image (cached EasyOCR reader)
    8.  Result stored thread-safely for frontend

    GPU optimisations
    ─────────────────
    • FP16 half-precision on seg + OBB
    • torch.no_grad() on all inference
    • verbose=False — no stdout per frame
    • Fixed imgsz — skips YOLO auto-resize
    • Frame skipping during gate phase
    • Model warmup on init
    • EasyOCR cached — loaded once, reused
    """

    def __init__(self, seg_model, obb_model, ocr_model=None, capture_duration=2):
        self.seg_model          = seg_model
        self.obb_model          = obb_model
        self.ocr_model          = ocr_model
        self.capture_duration   = capture_duration

        self.frame_buffer       = deque(maxlen=150)
        self.is_capturing       = False
        self.capture_start_time = None

        # Results — thread-safe
        self.results       = []
        self._results_lock = threading.Lock()

        # Gate stability tracking
        self._stable_history  = deque(maxlen=STABLE_FRAMES_NEEDED)
        self._gate_frame_tick = 0

        # Device
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INIT] Running on: {self._device.upper()}")

        # EasyOCR — init once, reuse (avoids 2-3s reload per tag)
        print("[INIT] Loading EasyOCR...")
        self._ocr_reader = easyocr.Reader(
            ['en'],
            gpu=(self._device == 'cuda'),
            verbose=False,
        )
        print("[INIT] EasyOCR ready")

        # Warmup
        self._warmup_models()

    # ══════════════════════════════════════════════════════════════════════ #
    #  WARMUP                                                                #
    # ══════════════════════════════════════════════════════════════════════ #
    def _warmup_models(self):
        print("[WARMUP] Warming up SEG and OBB models...")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        with torch.no_grad():
            for _ in range(2):
                self.seg_model(dummy, imgsz=SEG_IMGSZ,
                               half=(self._device == 'cuda'), verbose=False)
            for _ in range(2):
                self.obb_model(dummy,
                               half=(self._device == 'cuda'), verbose=False)
        print("[WARMUP] Done — models ready at full speed")

    # ══════════════════════════════════════════════════════════════════════ #
    #  SEGMENTATION                                                          #
    #                                                                        #
    #  Always returns a 3-tuple: (mask, corners, seg_result)                #
    #  seg_result is ALWAYS returned (even on no-detection) so the          #
    #  frontend can call seg_result.plot() to show live YOLO annotations.   #
    #  This is the pattern from the reference app.py snippet.               #
    # ══════════════════════════════════════════════════════════════════════ #
    def run_seg(self, frame: np.ndarray):
        """
        Returns (mask, corners, seg_result).
        mask     : H×W uint8 binary mask, or None
        corners  : (4,2) float32 rotated bbox from mask, or None
        seg_result: YOLO Results object — call .plot() for annotated frame
        """
        with torch.no_grad():
            results = self.seg_model(
                frame,
                conf=SEG_CONF_THRESHOLD,
                imgsz=SEG_IMGSZ,
                half=(self._device == 'cuda'),
                verbose=False,
            )

        seg_result = results[0]   # always keep this for .plot()

        # No detection — return result object so .plot() still renders cleanly
        if not (hasattr(seg_result, 'masks') and
                seg_result.masks is not None and
                len(seg_result.masks.data) > 0):
            return None, None, seg_result

        # Pick highest-confidence detection
        confs    = seg_result.boxes.conf.cpu().numpy() \
                   if seg_result.boxes is not None else None
        best_idx = int(confs.argmax()) \
                   if confs is not None and len(confs) > 0 else 0

        mask_tensor = seg_result.masks.data[best_idx].cpu().numpy()
        mask = (mask_tensor * 255).astype(np.uint8)

        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        corners = self._corners_from_mask(mask)
        return mask, corners, seg_result

    def _corners_from_mask(self, mask: np.ndarray):
        """
        4-point rotated bounding box from seg mask contour via minAreaRect.
        Same information as OBB corners but derived from the seg mask.
        Returns (4,2) float32 or None.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect)
        return corners.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════ #
    #  GATE CHECK                                                            #
    #                                                                        #
    #  ALWAYS returns a 4-tuple: (passed, mask, corners, seg_result)        #
    #  Every return path includes seg_result so the frontend always has      #
    #  a valid object to call .plot() on for the live annotated view.        #
    # ══════════════════════════════════════════════════════════════════════ #
    def check_tag_policy(self, frame: np.ndarray):
        """
        Returns (passed: bool, mask, corners, seg_result).
        All 4 values always present — seg_result is never omitted.
        """
        self._gate_frame_tick += 1

        # Frame skip — no inference this tick
        # Return None for seg_result so frontend keeps last overlay
        if self._gate_frame_tick % GATE_SKIP_FRAMES != 0:
            return False, None, None, None

        # Run seg — always get seg_result back
        mask, corners, seg_result = self.run_seg(frame)

        # No detection
        if mask is None or corners is None:
            self._stable_history.clear()
            return False, None, None, seg_result

        h, w       = frame.shape[:2]
        frame_area = h * w

        # Policy 1 — all 4 corners inside frame (no clipping)
        for x, y in corners:
            if x < 0 or y < 0 or x >= w or y >= h:
                self._stable_history.clear()
                return False, mask, corners, seg_result

        # Policy 2 — area sanity
        mask_area  = cv2.countNonZero(mask)
        area_ratio = mask_area / frame_area
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            self._stable_history.clear()
            return False, mask, corners, seg_result

        # Policy 3 — centroid stability
        M = cv2.moments(mask)
        if M["m00"] == 0:
            self._stable_history.clear()
            return False, mask, corners, seg_result

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        self._stable_history.append((cx, cy))

        if len(self._stable_history) < STABLE_FRAMES_NEEDED:
            return False, mask, corners, seg_result

        ref_cx, ref_cy = self._stable_history[0]
        for pcx, pcy in list(self._stable_history)[1:]:
            if (abs(pcx - ref_cx) > STABILITY_TOLERANCE or
                    abs(pcy - ref_cy) > STABILITY_TOLERANCE):
                return False, mask, corners, seg_result

        # All policies passed ✓
        return True, mask, corners, seg_result

    # ══════════════════════════════════════════════════════════════════════ #
    #  CAPTURE CONTROL                                                       #
    # ══════════════════════════════════════════════════════════════════════ #
    def start_capture(self):
        self.is_capturing       = True
        self.capture_start_time = datetime.now()
        self.frame_buffer.clear()
        self._stable_history.clear()
        self._gate_frame_tick   = 0
        print(f"[CAPTURE] Started at {self.capture_start_time.strftime('%H:%M:%S')}")

    def should_stop_capture(self) -> bool:
        if not self.is_capturing:
            return False
        elapsed = (datetime.now() - self.capture_start_time).total_seconds()
        return elapsed >= self.capture_duration

    # ══════════════════════════════════════════════════════════════════════ #
    #  SHARPNESS                                                             #
    # ══════════════════════════════════════════════════════════════════════ #
    def calc_sharpness(self, frame: np.ndarray,
                       mask: np.ndarray = None) -> float:
        region    = cv2.bitwise_and(frame, frame, mask=mask) \
                    if mask is not None else frame
        gray      = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        return lap_var * 0.7 + sobel_var * 0.3

    def get_sharpest_frame(self, buffer=None):
        src = buffer if buffer is not None else self.frame_buffer
        return max(src, key=lambda x: x['sharpness']) if src else None

    # ══════════════════════════════════════════════════════════════════════ #
    #  PERSPECTIVE TRANSFORM                                                 #
    # ══════════════════════════════════════════════════════════════════════ #
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        pts     = corners.reshape(4, 2)
        rect    = np.zeros((4, 2), dtype=np.float32)
        s       = pts.sum(axis=1)
        diff    = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def apply_perspective_transform(self, frame: np.ndarray,
                                    corners: np.ndarray) -> np.ndarray:
        rect           = self._order_corners(corners)
        tl, tr, br, bl = rect
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        dst   = np.array([[0, 0], [out_w-1, 0],
                          [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (out_w, out_h))

    # ══════════════════════════════════════════════════════════════════════ #
    #  OCR                                                                   #
    # ══════════════════════════════════════════════════════════════════════ #
    def extract_text(self, image: np.ndarray) -> str:
        start  = time.time()
        texts  = self._ocr_reader.readtext(image, detail=0)
        result = " ".join(texts).lower()
        print(f"[OCR] {time.time()-start:.2f}s → '{result}'")
        return result

    # ══════════════════════════════════════════════════════════════════════ #
    #  MAIN PROCESSING                                                       #
    # ══════════════════════════════════════════════════════════════════════ #
    def process_sharpest_frame(self, best: dict) -> dict:
        """
        1. Perspective warp using SEG corners
        2. OBB on warped image — .plot() gives annotated frame
        3. OCR on warped image
        4. Store and return result dict
        """
        frame   = best['frame']
        mask    = best['mask']
        corners = best['corners']

        print("[PROCESS] Warp → OBB → OCR")

        # Step 1 — warp
        if corners is not None:
            warped = self.apply_perspective_transform(frame, corners)
        elif mask is not None:
            x, y, w, h = cv2.boundingRect(mask)
            warped = frame[y:y+h, x:x+w]
        else:
            warped = frame

        # Step 2 — OBB on warped (.plot() draws boxes + labels)
        obb_corners   = None
        obb_annotated = None
        with torch.no_grad():
            obb_results = self.obb_model(
                warped,
                half=(self._device == 'cuda'),
                verbose=False,
            )
        if obb_results:
            r_obb         = obb_results[0]
            obb_annotated = r_obb.plot()   # annotated warped image
            if (hasattr(r_obb, 'obb') and
                    r_obb.obb is not None and
                    len(r_obb.obb.xyxyxyxy) > 0):
                obb_corners = r_obb.obb.xyxyxyxy[0].cpu().numpy().tolist()

        # Step 3 — OCR
        ocr_text = self.extract_text(warped)

        result = {
            'id'           : len(self.results),
            'timestamp'    : datetime.now().isoformat(),
            'frame'        : frame,           # original sharpest frame
            'mask'         : mask,
            'warped_frame' : warped,          # plain perspective-corrected tag
            'obb_annotated': obb_annotated,   # warped + OBB boxes from .plot()
            'seg_corners'  : corners.tolist() if corners is not None else None,
            'obb_corners'  : obb_corners,
            'ocr_result'   : ocr_text,
            'sharpness'    : best['sharpness'],
        }

        with self._results_lock:
            self.results.append(result)

        print(f"[RESULT] #{result['id']} | sharp={result['sharpness']:.1f}"
              f" | ocr='{ocr_text}'")
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