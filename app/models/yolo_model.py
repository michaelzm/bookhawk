import logging
from ultralytics import YOLO
from app.config.settings import YOLO_MODEL_PATH

def load_yolo():
    """Initialize and load the YOLO model."""
    logging.info("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    logging.info("YOLO model loaded successfully.")
    return model