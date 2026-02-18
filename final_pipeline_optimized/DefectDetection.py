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
STABLE_FRAMES_NEEDED = 5      # consecutive clean SEG frames before capture
STABILITY_TOLERANCE  = 15     # max centroid pixel shift still considered stable
MIN_AREA_RATIO       = 0.02   # tag mask must cover at least 2% of frame
MAX_AREA_RATIO       = 0.90   # tag mask must cover at most 90% of frame
SEG_CONF_THRESHOLD   = 0.55   # raise if seg picks up background/faces
SEG_IMGSZ            = 640    # fixed input size → skips per-frame resize logic
GATE_SKIP_FRAMES     = 3      # run seg every Nth frame during gate phase


class DefectDetectionOCR:
    """
    GPU-Optimised Pipeline
    ──────────────────────
    1.  SEG model watches every Nth frame (gate phase, FP16, verbose off)
    2.  Gate: tag fully inside frame + stable centroid for N frames
    3.  Once gate passes → 2-second capture buffer (seg runs every frame)
    4.  Sharpest frame selected from buffer
    5.  Perspective warp from SEG mask corners (minAreaRect)
    6.  OBB model on warped image (FP16, verbose off)
    7.  OCR on warped image (EasyOCR GPU)
    8.  Result stored thread-safely for frontend

    GPU optimisations applied
    ─────────────────────────
    • FP16 half-precision on seg + OBB (RTX tensor cores, ~2x speedup)
    • torch.no_grad() on all inference (no gradient tracking overhead)
    • verbose=False (no stdout per frame, small but free saving)
    • Fixed imgsz (skips YOLO auto-resize detection every frame)
    • Frame skipping during gate (seg runs every GATE_SKIP_FRAMES frames)
    • Model warmup on init (CUDA kernel compile done before first real frame)
    • CAP_PROP_BUFFERSIZE=1 handled in frontend (always freshest frame)
    • EasyOCR initialised once and cached (not re-created per inference)
    """

    def __init__(self, seg_model, obb_model, ocr_model=None, capture_duration=2):
        self.seg_model        = seg_model
        self.obb_model        = obb_model
        self.ocr_model        = ocr_model
        self.capture_duration = capture_duration

        self.frame_buffer       = deque(maxlen=150)
        self.is_capturing       = False
        self.capture_start_time = None
        self.processing_queue   = queue.Queue()

        # Results — thread-safe
        self.results       = []
        self._results_lock = threading.Lock()

        # Gate stability
        self._stable_history  = deque(maxlen=STABLE_FRAMES_NEEDED)
        self._gate_frame_tick = 0   # counts frames for skip logic

        # GPU device
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

        # Warmup both models so first real frame is fast
        self._warmup_models()

    # ══════════════════════════════════════════════════════════════════════ #
    #  WARMUP                                                                #
    # ══════════════════════════════════════════════════════════════════════ #
    def _warmup_models(self):
        """
        Run both models twice on a dummy blank frame.
        Forces CUDA to compile kernels + allocate memory upfront
        so first real frame inference is at full speed.
        """
        print("[WARMUP] Warming up SEG and OBB models...")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        with torch.no_grad():
            for _ in range(2):
                self.seg_model(
                    dummy,
                    imgsz=SEG_IMGSZ,
                    half=(self._device == 'cuda'),
                    verbose=False,
                )
            for _ in range(2):
                self.obb_model(
                    dummy,
                    half=(self._device == 'cuda'),
                    verbose=False,
                )
        print("[WARMUP] Done — models ready at full speed")

    # ══════════════════════════════════════════════════════════════════════ #
    #  SEGMENTATION                                                          #
    # ══════════════════════════════════════════════════════════════════════ #
    def run_seg(self, frame: np.ndarray):
        """
        GPU-optimised seg inference.
        Returns (mask_uint8, corners_4x2, results[0]) or (None, None, None).

        results[0] is returned so the frontend can call .plot() directly —
        YOLO's built-in renderer draws masks, boxes, labels cleanly in one call.

        Optimisations:
          • half=True  → FP16 on CUDA tensor cores
          • imgsz fixed → no per-frame resize detection
          • verbose=False → no stdout overhead
          • torch.no_grad() → no gradient tracking
        """
        with torch.no_grad():
            results = self.seg_model(
                frame,
                conf=SEG_CONF_THRESHOLD,
                imgsz=SEG_IMGSZ,
                half=(self._device == 'cuda'),
                verbose=False,
            )

        r = results[0]

        if not (hasattr(r, 'masks') and
                r.masks is not None and
                len(r.masks.data) > 0):
            # Return the result object even on no-detection so .plot() still works
            return None, None, r

        # Pick highest-confidence detection
        confs    = r.boxes.conf.cpu().numpy() \
                   if r.boxes is not None else None
        best_idx = int(confs.argmax()) \
                   if confs is not None and len(confs) > 0 else 0

        mask_tensor = r.masks.data[best_idx].cpu().numpy()
        mask = (mask_tensor * 255).astype(np.uint8)

        # Resize mask to original frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        corners = self._corners_from_mask(mask)
        return mask, corners, r

    def _corners_from_mask(self, mask: np.ndarray):
        """
        Derives the 4-point rotated bounding box from the seg mask contour
        using minAreaRect + boxPoints — same result as OBB but from seg mask.
        Returns (4,2) float32 or None.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect)
        return corners.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════ #
    #  GATE CHECK                                                            #
    # ══════════════════════════════════════════════════════════════════════ #
    def check_tag_policy(self, frame: np.ndarray):
        """
        Gate check with frame skipping:
          - Runs seg only every GATE_SKIP_FRAMES frames (saves 66% GPU time)
          - Policies: corners inside frame + area ratio + centroid stability
        Returns (passed: bool, mask, corners)
        """
        self._gate_frame_tick += 1

        # Skip frames — return last known state without running inference
        if self._gate_frame_tick % GATE_SKIP_FRAMES != 0:
            return False, None, None

        mask, corners, seg_result = self.run_seg(frame)

        if mask is None or corners is None:
            self._stable_history.clear()
            return False, None, None, seg_result

        h, w       = frame.shape[:2]
        frame_area = h * w

        # Policy 1 — all 4 corners inside frame (no clipping)
        for x, y in corners:
            if x < 0 or y < 0 or x >= w or y >= h:
                self._stable_history.clear()
                return False, None, None

        # Policy 2 — area sanity check
        mask_area  = cv2.countNonZero(mask)
        area_ratio = mask_area / frame_area
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            self._stable_history.clear()
            return False, None, None, seg_result

        # Policy 3 — centroid stability across N frames
        M = cv2.moments(mask)
        if M["m00"] == 0:
            self._stable_history.clear()
            return False, None, None, seg_result

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        self._stable_history.append((cx, cy))

        if len(self._stable_history) < STABLE_FRAMES_NEEDED:
            return False, None, None, seg_result

        ref_cx, ref_cy = self._stable_history[0]
        for pcx, pcy in list(self._stable_history)[1:]:
            if (abs(pcx - ref_cx) > STABILITY_TOLERANCE or
                    abs(pcy - ref_cy) > STABILITY_TOLERANCE):
                return False, None, None, seg_result

        return True, mask, corners, seg_result

    # ══════════════════════════════════════════════════════════════════════ #
    #  CAPTURE CONTROL                                                       #
    # ══════════════════════════════════════════════════════════════════════ #
    def start_capture(self):
        self.is_capturing       = True
        self.capture_start_time = datetime.now()
        self.frame_buffer.clear()
        self._stable_history.clear()
        self._gate_frame_tick = 0
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
        """Laplacian + Sobel combined sharpness — runs on CPU (fast enough)."""
        region    = cv2.bitwise_and(frame, frame, mask=mask) \
                    if mask is not None else frame
        gray      = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
        return lap_var * 0.7 + sobel_var * 0.3

    # Alias so old code still works
    def Sharpest_frame(self, frame: np.ndarray,
                       mask: np.ndarray = None) -> float:
        return self.calc_sharpness(frame, mask)

    def get_sharpest_frame(self, buffer=None):
        src = buffer if buffer is not None else self.frame_buffer
        return max(src, key=lambda x: x['sharpness']) if src else None

    # ══════════════════════════════════════════════════════════════════════ #
    #  PERSPECTIVE TRANSFORM                                                 #
    # ══════════════════════════════════════════════════════════════════════ #
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort 4 pts → [top-left, top-right, bottom-right, bottom-left]."""
        pts     = corners.reshape(4, 2)
        rect    = np.zeros((4, 2), dtype=np.float32)
        s       = pts.sum(axis=1)
        diff    = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left
        rect[2] = pts[np.argmax(s)]     # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def apply_perspective_transform(self, frame: np.ndarray,
                                    corners: np.ndarray) -> np.ndarray:
        """Warp tag to flat front-facing rectangle using SEG corners."""
        rect        = self._order_corners(corners)
        tl, tr, br, bl = rect
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        dst   = np.array(
            [[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (out_w, out_h))

    # ══════════════════════════════════════════════════════════════════════ #
    #  OCR                                                                   #
    # ══════════════════════════════════════════════════════════════════════ #
    def extract_text(self, image: np.ndarray) -> str:
        """
        Run EasyOCR using the cached reader (no reload overhead).
        GPU-accelerated when CUDA is available.
        """
        start  = time.time()
        texts  = self._ocr_reader.readtext(image, detail=0)
        result = " ".join(texts).lower()
        print(f"[OCR] {time.time()-start:.2f}s → '{result}'")
        return result

    # ══════════════════════════════════════════════════════════════════════ #
    #  MAIN PROCESSING  (runs once after capture ends)                      #
    # ══════════════════════════════════════════════════════════════════════ #
    def process_sharpest_frame(self, best: dict) -> dict:
        """
        1. Perspective warp using SEG mask corners
        2. OBB on warped image (FP16, verbose off, no_grad)
        3. OCR on warped image (cached EasyOCR reader)
        4. Save result
        """
        frame   = best['frame']
        mask    = best['mask']
        corners = best['corners']

        print("[PROCESS] Warp → OBB → OCR")

        # Step 1 — perspective warp
        if corners is not None:
            warped = self.apply_perspective_transform(frame, corners)
        else:
            if mask is not None:
                x, y, w, h = cv2.boundingRect(mask)
                warped = frame[y:y+h, x:x+w]
            else:
                warped = frame

        # Step 2 — OBB on warped (FP16 + no_grad + verbose off)
        obb_corners    = None
        obb_annotated  = None   # warped frame with OBB boxes drawn via .plot()
        with torch.no_grad():
            obb_results = self.obb_model(
                warped,
                half=(self._device == 'cuda'),
                verbose=False,
            )
        if obb_results:
            r_obb = obb_results[0]
            # .plot() draws OBB boxes + labels on the warped image — clean one-liner
            obb_annotated = r_obb.plot()
            if (hasattr(r_obb, 'obb') and
                    r_obb.obb is not None and
                    len(r_obb.obb.xyxyxyxy) > 0):
                obb_corners = r_obb.obb.xyxyxyxy[0].cpu().numpy().tolist()

        # Step 3 — OCR
        ocr_text = self.extract_text(warped)

        result = {
            'id'          : len(self.results),
            'timestamp'   : datetime.now().isoformat(),
            'frame'       : frame,
            'mask'        : mask,
            'warped_frame': warped,
            'seg_corners' : corners.tolist() if corners is not None else None,
            'obb_corners' : obb_corners,
            'ocr_result'  : ocr_text,
            'sharpness'   : best['sharpness'],
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