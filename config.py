# Configuration for Sign Language Production

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "gesture_classifier_v1.h5"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "model_complexity": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}

# Gesture configuration
GESTURES = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "good",
    "bad",
    "love",
    "peace",
    "stop",
    "ok",
    "help",
    "sorry",
]

# Training configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "test_size": 0.1,
}

# Video configuration
VIDEO_CONFIG = {
    "frame_width": 1280,
    "frame_height": 720,
    "fps": 30,
    "flip_frame": True,
}

# App configuration
APP_CONFIG = {
    "window_title": "Sign Language Production (SLP)",
    "window_width": 1280,
    "window_height": 720,
    "theme": "dark",
}

# Data collection
DATA_COLLECTION_CONFIG = {
    "samples_per_gesture": 100,
    "frames_per_sample": 30,
    "capture_fps": 30,
}
