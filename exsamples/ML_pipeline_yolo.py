import cv2
import time 
import numpy as np
from ultralytics import YOLO

# Load your models here
seg_model = YOLO("yolo11n-seg.pt") # Detection/Segmentation
cls_model = YOLO("yolo11m-cls.pt") # Defect Classification

def get_sharpness(frame):
    """Combines Laplacian and Sobel for a highly accurate focus score."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = np.sqrt(sobelx**2 + sobely**2).var()

    sharpest_frame=laplacian_var * 0.7 + sobel_var * 0.3
    # Weighted average: Sobel detects edges, Laplacian detects fine detail
    return sharpest_frame

def extract_nd_wrap(frame, box_points, rect):
    """Straightens the tilted object for better classification accuracy."""
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    src_pts = box_points.astype("float32")
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, matrix, (width, height))

# SETTINGS
BUFFER_TIME = 2.0  
cap = cv2.VideoCapture(0)
frame_buffer = []
active_collection = False
start_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run segmentation
    results = seg_model(frame, conf=0.7, verbose=False)[0]

    if results.masks is not None:
        if not active_collection:
            active_collection = True
            start_time = time.time()
            frame_buffer = [] # Clear buffer for new detection
            print("Object detected. Buffering...")

        # Collect frames during the 2-second window
        if time.time() - start_time <= BUFFER_TIME:
            score = get_sharpness(frame)
            frame_buffer.append({
                "score": score,
                "frame": frame.copy(),
                "mask": results.masks.data[0].cpu().numpy().astype(np.uint8)
            })
        else:
            # WINDOW FINISHED: Process the best frame
            if frame_buffer:
                sharpest = max(frame_buffer, key=lambda x: x['score']) 
                contours, _ = cv2.findContours(sharpest['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                 
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    box = np.int32(cv2.boxPoints(rect)) # Fixed np.int0
                    
                    # Warp and Classify
                    extracted_img = extract_nd_wrap(sharpest['frame'], box, rect)
                    cls_results = cls_model.predict(extracted_img, verbose=False)[0]
                    
                    # Classification models use .probs, not .boxes
                    prediction = cls_results.names[cls_results.probs.top1]
                    confidence = cls_results.probs.top1conf.item()

                    print(f"RESULT: {prediction.upper()} ({confidence:.2%})")
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    cv2.putText(frame, f"{prediction} ({confidence:.2%})", (box[0][0], box[0][1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Extracted Object", extracted_img)
                    cv2.waitKey(2000) 
            
            # CRITICAL FIX: Reset outside the mask check to allow re-triggering
            active_collection = False 

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()