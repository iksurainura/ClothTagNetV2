import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time


class SharpFrameClassifier:

    def __init__(self, model_path, capture_duration=2.0):
        self.capture_duration = capture_duration
        self.model = YOLO(model_path)
    
    def calculate_sharpness(self):
        gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        laplacian_var=cv2.Laplacian(gray,cv2.CV_64F).var()

        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
        sobel_var=np.sqrt(sobelx**2+sobely**2).var()

        sharpness=laplacian_var * 0.7 + sobel_var * 0.3

        return sharpness
    
    def capture_segment(self,camera_index=0):
        """Capture video frames from the camera for a specified duration."""

        #change the 0 to the desired camera index
        cap=cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_index}")
        
        #actual video frame rate
        camera_fps=cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {camera_fps}")

        frames=[]
        start_time=time.time()

        while (time.time()-start_time)<self.capture_duration:
            ret,self.frame=cap.read()
            if not ret:
                break
            frames.append(self.frame)

        cap.release()
        print("Video capture complete.")
        print("Total video duration: {:.2f} seconds".format(time.time()-start_time))
        print(f"Total frames captured: {len(frames)}")
        return frames
    
    def get_sharpest_frames(self, frames, top_k=5):
        if not frames:
            print("No frames to process.")
            return []
        # Go through each frame and calculate sharpness scores
        scores = [(self.calculate_sharpness(f), i) for i, f in enumerate(frames)]
        # Get indices of top_k sharpest frames
        top_indices = np.argpartition([s[0] for s in scores], -top_k)[-top_k:]
        # Return the top_k sharpest frames
        sharpest_frames = [frames[i] for i in top_indices]
        return sharpest_frames
    

    
    def classify_frames(self, frames):
        sharpest_frames = self.get_sharpest_frames(frames)
        result=[]
        for frame in sharpest_frames:
            resized_frame=cv2.resize(frame,(640,640))
            result=self.model.predict(resized_frame,conf=0.5,save=False,verbose=False)
            if hasattr(result,'probs'):
                class_id=result.probs.top1
                confidence=result.probs.top1conf.item()
                class_name=self.names[class_id]
            print(f"Class: {class_names}, Confidence: {conf:.2f}")
            
        coloured_frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        result=self.model.predict(self.get_sharpest_frames(frames=self.capture_segment),conf=0.5,save=False)
        print(result.obb.cls)
        print(result)