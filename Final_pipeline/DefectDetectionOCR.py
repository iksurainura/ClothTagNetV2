import torch
import numpy as np
from collections import deque
from datetime import datetime
import threading
import queue
import cv2
import easyocr
import time
import warnings

# ── Tunable constants ──────────────────────────────────────────────────────
STABLE_FRAMES_NEEDED = 5
STABILITY_TOLERANCE  = 15       # pixels
MIN_AREA_RATIO       = 0.015    # slightly lowered — helps with small/distant tags
MAX_AREA_RATIO       = 0.92
SEG_CONF_THRESHOLD   = 0.57     # small increase — reduces false positives

# Recommended: create EasyOCR reader once (heavy initialization)
# We'll do lazy initialization in __init__ or first use
GLOBAL_OCR_READER = None


class DefectDetectionOCR:
    def __init__(self, seg_model, obb_model, capture_duration=2.0):
        self.seg_model = seg_model
        self.obb_model = obb_model
        self.capture_duration = capture_duration

        self.frame_buffer     = deque(maxlen=180)   # ~6s @ 30 fps — safer headroom
        self.is_capturing     = False
        self.capture_start_time = None
        self.processing_queue = queue.Queue()       # currently unused — future async option

        self.results       = []
        self._results_lock = threading.Lock()

        self._stable_history = deque(maxlen=STABLE_FRAMES_NEEDED)

        # Lazy OCR initialization (GPU check + language once)
        self._ocr_reader = None

    # ─────────────────────────────────────────────────────────────
    #   Lazy OCR reader (recommended pattern)
    # ─────────────────────────────────────────────────────────────
    @property
    def ocr_reader(self):
        global GLOBAL_OCR_READER
        if GLOBAL_OCR_READER is not None:
            self._ocr_reader = GLOBAL_OCR_READER
        if self._ocr_reader is None:
            print("[OCR] Initializing EasyOCR (this may take several seconds)...")
            try:
                self._ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
                GLOBAL_OCR_READER = self._ocr_reader  # share across instances if multiple
            except Exception as e:
                warnings.warn(f"EasyOCR initialization failed: {e}\nFalling back to CPU.")
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                GLOBAL_OCR_READER = self._ocr_reader
        return self._ocr_reader

    # ─────────────────────────────────────────────────────────────
    #   SEGMENTATION & CORNERS
    # ─────────────────────────────────────────────────────────────
    def run_seg(self, frame: np.ndarray):
        # Add try-except because YOLO can sometimes raise on corrupt frames
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

        # Select best mask by confidence × area (more robust than conf alone)
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            areas = (results[0].boxes.xywh[:,2] * results[0].boxes.xywh[:,3]).cpu().numpy()
            scores = confs * areas
            best_idx = scores.argmax()
        else:
            best_idx = 0

        mask_tensor = masks[best_idx].cpu().numpy()
        mask = (mask_tensor > 0.5).astype(np.uint8) * 255   # binarize properly

        # Match original size if model gave different resolution
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        corners = self._corners_from_mask(mask)
        return mask, corners

    def _corners_from_mask(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Use largest contour — but add minimum area filter
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 150:  # arbitrary small area protection
            return None

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    #   GATE CHECK  (unchanged logic — just cleaner)
    # ─────────────────────────────────────────────────────────────
    def check_tag_policy(self, frame: np.ndarray):
        mask, corners = self.run_seg(frame)
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

    # ─────────────────────────────────────────────────────────────
    #   CAPTURE CONTROL
    # ─────────────────────────────────────────────────────────────
    def start_capture(self):
        self.is_capturing = True
        self.capture_start_time = datetime.now()
        self.frame_buffer.clear()
        self._stable_history.clear()
        print(f"[CAPTURE START] {self.capture_start_time.strftime('%H:%M:%S.%f')[:-3]}")

    def should_stop_capture(self) -> bool:
        if not self.is_capturing:
            return False
        elapsed = (datetime.now() - self.capture_start_time).total_seconds()
        return elapsed >= self.capture_duration

    # ─────────────────────────────────────────────────────────────
    #   SHARPNESS (small improvement — more stable score)
    # ─────────────────────────────────────────────────────────────
    def calc_sharpness(self, frame: np.ndarray, mask: np.ndarray = None) -> float:
        if mask is not None and cv2.countNonZero(mask) > 100:
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            masked = frame

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # Combine three common sharpness estimators → more robust ranking
        lap   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1).var()
        tenengrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0).var() + cv2.Sobel(gray, cv2.CV_64F, 0, 1).var()
        return 0.5 * lap + 0.3 * sobel + 0.2 * tenengrad

    def get_sharpest_frame(self, buffer=None):
        buf = buffer if buffer is not None else self.frame_buffer
        if not buf:
            return None
        return max(buf, key=lambda x: x.get('sharpness', 0))

    # ─────────────────────────────────────────────────────────────
    #   PERSPECTIVE WARP
    # ─────────────────────────────────────────────────────────────
    def apply_perspective_transform(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if corners is None:
            return frame

        rect = self._order_corners(corners)
        tl, tr, br, bl = rect

        width_a  = np.linalg.norm(tr - tl)
        width_b  = np.linalg.norm(br - bl)
        height_a = np.linalg.norm(bl - tl)
        height_b = np.linalg.norm(br - tr)

        out_w = max(int(width_a),  int(width_b))
        out_h = max(int(height_a), int(height_b))

        # Protection against degenerate transforms
        if out_w < 16 or out_h < 16:
            return frame

        dst = np.array([
            [0,      0     ],
            [out_w-1, 0     ],
            [out_w-1, out_h-1],
            [0,      out_h-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        return warped

    # ─────────────────────────────────────────────────────────────
    #   OCR
    # ─────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────
    #   MAIN PROCESSING
    # ─────────────────────────────────────────────────────────────
    def process_sharpest_frame(self, best: dict) -> dict:
        frame   = best['frame']
        mask    = best.get('mask')
        corners = best.get('corners')

        print("[PROCESS] warp → obb → ocr")

        # Warp
        warped = self.apply_perspective_transform(frame, corners)

        # OBB
        try:
            obb_results = self.obb_model(warped, verbose=False)
            obb_corners = None
            if obb_results and obb_results[0].obb is not None:
                first_obb = obb_results[0].obb
                if len(first_obb.xyxyxyxy) > 0:
                    obb_corners = first_obb.xyxyxyxy[0].cpu().numpy().tolist()
        except Exception as e:
            print(f"[OBB] inference failed: {e}")
            obb_corners = None

        # OCR
        ocr_text = self.extract_text(warped)

        result = {
            'id'           : len(self.results),
            'timestamp'    : datetime.now().isoformat(),
            'frame'        : frame,
            'mask'         : mask,
            'warped_frame' : warped,
            'seg_corners'  : corners.tolist() if corners is not None else None,
            'obb_corners'  : obb_corners,
            'ocr_result'   : ocr_text,
            'sharpness'    : best.get('sharpness', 0.0),
        }

        with self._results_lock:
            self.results.append(result)

        print(f"[RESULT #{result['id']}] sharpness={result['sharpness']:.1f} | ocr='{ocr_text}'")
        return result

    # ─────────────────────────────────────────────────────────────
    #   FRONTEND HELPERS (unchanged)
    # ─────────────────────────────────────────────────────────────
    def get_all_results(self):
        with self._results_lock:
            return list(self.results)

    def get_latest_result(self):
        with self._results_lock:
            return self.results[-1] if self.results else None

    def clear_results(self):
        with self._results_lock:
            self.results.clear()