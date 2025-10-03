import logging
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.models.yolo_model import load_yolo
from app.models.sam_model import load_sam
from app.routes.book_routes import create_book_router

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Determine device
if torch.backends.mps.is_available():  # Apple M1 GPU
    device = torch.device("mps")
elif torch.cuda.is_available():        # Nvidia GPU
    device = torch.device("cuda")
else:
    device = torch.device("cpu")       # fallback to CPU

logging.info(f"Using device: {device}")

# Load models
yolo_model = load_yolo()
sam_predictor = load_sam(device)

# Include routers
app.include_router(create_book_router(yolo_model, sam_predictor))