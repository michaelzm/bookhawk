from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import logging
import io
from PIL import Image
import numpy as np
import json
import asyncio
import os

# Import your services and models properly
from app.services.ocr_service import ocr_from_file
from app.services.image_service import (
    process_yolo_detection,
    process_sam_segmentation,
    encode_image_base64
)
from app.config.settings import ABORT_BOOK_COUNT

router = APIRouter()

def create_book_router(yolo_model, sam_predictor):
    """
    Factory function to create book router with injected dependencies
    """
    
    @router.get("/", response_class=HTMLResponse)
    def index():
        """Serve the index page."""
        logging.info("Serving index.html")
        try:
            with open("static/index.html") as f:
                return f.read()
        except FileNotFoundError:
            return "<h1>Index page not found</h1>"
    
    @router.post("/upload")
    async def upload_image(file: UploadFile = File(...)):
        """Handle image upload and processing."""
        logging.info(f"Upload endpoint called with file: {file.filename}")
        content = await file.read()
        
        async def process_and_stream():
            try:
                # Decode image
                logging.info("Reading and decoding image.")
                image = Image.open(io.BytesIO(content)).convert("RGB")
                img_np = np.array(image)
                logging.info("Image decoded successfully.")
                
                # Run YOLO detection
                logging.info("Starting YOLO detection.")
                det_results = yolo_model(img_np)
                img_with_boxes, box_infos = process_yolo_detection(img_np, det_results)
                
                # Stream initial image with boxes
                initial_b64 = encode_image_base64(img_with_boxes)

                shelf_result = {
                    "shelf_with_boxes_b64": initial_b64
                }
                yield json.dumps({"type": "initial", "data": shelf_result}) + "\n"
                await asyncio.sleep(0.1)
                logging.info(f"YOLO detection found {len(box_infos)} books.")
                
                if not box_infos:
                    logging.warning("No books found in image.")
                    yield json.dumps({"type": "error", "message": "No books found in image."}) + "\n"
                    return
                
                # Process each detected book
                books = []
                for i, box_info in enumerate(box_infos):
                    if i >= 0:  # Limit to 3 books
                        logging.warning("Aborting due to too many books")
                        break
                    
                    logging.info(f"Processing book {i+1}/{len(box_infos)}.")
                    bbox = box_info["box"]
                    prob = box_info["prob"]
                    
                    # Get YOLO crop
                    x1, y1, x2, y2 = map(int, bbox)
                    yolo_crop = img_np[y1:y2, x1:x2]
                    yolo_b64 = encode_image_base64(yolo_crop)
                    
                    # Process SAM segmentation
                    logging.info(f"Running SAM segmentation for book {i+1}.")
                    segmented_image, _, _, _ = process_sam_segmentation(
                        img_np, sam_predictor, bbox
                    )
                    if segmented_image is None:
                        continue
                    
                    # Save temporary file for OCR (you might want to avoid this)
                    temp_path = f"temp_upload_{hash(tuple(bbox))}.png"
                    Image.fromarray(segmented_image).save(temp_path)
                    seg_b64 = encode_image_base64(segmented_image)
                    
                    # Perform OCR
                    logging.info(f"Performing OCR for book {i+1}.")
                    ocr_result = ocr_from_file(temp_path)
                    logging.info(f"OCR for book {i+1} completed.")
                    
                    # Clean up temporary file
                    os.remove(temp_path)
                    
                    # Create book result
                    book_result = {
                        "id": i,
                        "yolo_box": bbox,
                        "yolo_prob": prob,
                        "yolo_crop_b64": yolo_b64,
                        "sam_segmentation_b64": seg_b64,
                        "ocr_result": ocr_result
                    }
                    books.append(book_result)
                    yield json.dumps({"type": "book", "data": book_result}) + "\n"
                    await asyncio.sleep(0.1)
                
                # Send final result with all books
                yield json.dumps({"type": "complete", "books": books}) + "\n"
                
            except Exception as e:
                logging.error(f"Error processing and streaming image: {e}", exc_info=True)
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        
        headers = {
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-ndjson'
        }
        
        return StreamingResponse(
            process_and_stream(),
            media_type="application/x-ndjson",
            headers=headers
        )
    
    return router