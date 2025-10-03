import torch
import logging
from segment_anything import SamPredictor, sam_model_registry
from app.config.settings import SAM_MODEL_PATH

def load_sam(device):
    """Initialize and load the SAM model."""
    logging.info("Loading SAM model...")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(SAM_MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Initialize the model
        sam = sam_model_registry["vit_b"](checkpoint=None)
        
        # Load state dict and move to device
        sam.load_state_dict(state_dict)
        sam.to(device)

        # Create the predictor
        predictor = SamPredictor(sam)
        logging.info("SAM model loaded successfully.")
        return predictor
        
    except Exception as e:
        logging.error(f"Error loading SAM model: {str(e)}")
        raise