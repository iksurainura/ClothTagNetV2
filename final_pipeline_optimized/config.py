# config.py
"""
Configuration settings for the inspection system
"""

CONFIG = {
    # Model paths - update these to your actual model paths
    'seg_model_path': 'path/to/seg_model.pt',
    'obb_model_path': 'path/to/obb_model.pt',
    
    # Video source: 0 for webcam, or path to video file / RTSP stream
    'video_source': '0',
    
    # Detection parameters
    'capture_duration': 2.0,      # seconds to capture after trigger
    'min_area_ratio': 0.015,      # minimum tag area relative to frame
    'max_area_ratio': 0.92,       # maximum tag area relative to frame
    'seg_conf_threshold': 0.57,   # segmentation confidence threshold
    'stability_tolerance': 15,    # pixels of movement allowed for stability
    'stable_frames_needed': 5,    # consecutive stable frames required
    
    # Processing
    'use_gpu': True,              # use CUDA if available
    'buffer_size': 180,           # max frames in buffer (~6s @ 30fps)
    
    # Display
    'show_fps': True,
    'save_results': False,
    'output_dir': './inspection_results'
}