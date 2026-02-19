# video_stream.py
"""
Threaded video capture for non-blocking frame acquisition
"""

import cv2
import threading
import time
from collections import deque


class VideoStream:
    def __init__(self, source=0, buffer_size=30):
        self.source = source
        self.buffer_size = buffer_size
        self.stream = None
        # Use deque for fixed-size buffer instead of Queue (Queue doesn't support maxlen)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.stopped = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        
    def start(self):
        """Start the video capture thread"""
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
            
        self.stream = cv2.VideoCapture(self.source)
        
        # Set buffer size to reduce latency
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.stream.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self
    
    def _update(self):
        """Continuously grab frames from stream"""
        while not self.stopped:
            if not self.stream.isOpened():
                time.sleep(0.01)
                continue
                
            ret, frame = self.stream.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            with self.lock:
                self.latest_frame = frame
            
            # Add to deque buffer (thread-safe with lock)
            with self.buffer_lock:
                self.frame_buffer.append(frame)
    
    def read(self):
        """Get the latest frame"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def read_buffered(self):
        """Get oldest frame from buffer (FIFO)"""
        with self.buffer_lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer.popleft()
            return None
    
    def get_buffer_size(self):
        """Get current buffer fill level"""
        with self.buffer_lock:
            return len(self.frame_buffer)
    
    def stop(self):
        """Stop the video stream"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.stream is not None:
            self.stream.release()
    
    def is_active(self):
        return self.stream is not None and self.stream.isOpened() and not self.stopped
    
    def get_fps(self):
        if self.stream:
            return self.stream.get(cv2.CAP_PROP_FPS)
        return 0
    
    def get_resolution(self):
        if self.stream:
            w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)