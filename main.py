import os
import base64
import logging
logging.basicConfig(level=logging.INFO)
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import json
import asyncio
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import requests
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.178.39:11434")
MODEL = os.getenv("MODEL", "qwen2.5vl:7b")
PROMPT = os.getenv("PROMPT", "You will receive a image of a book front view in a book shelf. Extract the author and title of the book image. Return the output as JSON. Output the JSON as {'author': .. , 'title': .., 'language': ..}. Do not mistake the genre or publisher as the author. If you cannot extract the author from the image, output the author that has published the book that you know of.")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov10x.pt")
BOOK_CLASS_INDEX = int(os.getenv("BOOK_CLASS_INDEX", "73"))
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "sam2.1_b.pt")
ABORT_BOOK_COUNT = int(os.getenv("ABORT_BOOK_COUNT", "30"))


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

logging.info("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
logging.info("YOLO model loaded.")
logging.info("Loading SAM model...")

# Path to your SAM checkpoint (use the Base model for M1: vit_b)
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"

# Load the checkpoint and unwrap if necessary
checkpoint = torch.load(SAM_MODEL_PATH, map_location="mps")  # M1 GPU
if "model" in checkpoint:
    checkpoint = checkpoint["model"]

# Build the SAM model without passing checkpoint
sam = sam_model_registry["vit_b"](checkpoint=None)

# Load the state_dict manually
sam.load_state_dict(checkpoint)
sam.to("mps")  # run on Apple GPU

# Create the predictor
sam_predictor = SamPredictor(sam)

# Create the predictor
sam_predictor = SamPredictor(sam)
logging.info("SAM model loaded.")

def ocr(ollama_host, model_name, prompt, image_path):
    logging.info(f"Connecting to Ollama host at {ollama_host} with model {model_name}")
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image file not found at '{image_path}'")
            return {"error": f"Image file not found at '{image_path}'"}
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [encoded_image]
            }],
            "stream": False
        }
        logging.info("Sending request to Ollama API.")
        response = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        logging.info("Ollama API request successful.")
        return response.json()['message']['content']
    except Exception as e:
        logging.error(f"Error in OCR function: {e}")
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index():
    logging.info("Serving index.html")
    with open("static/index.html") as f:
        return f.read()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    logging.info("Upload endpoint called with file: " + file.filename)
    
    # Read the file content immediately to prevent "seek of closed file" error
    content = await file.read()

    async def process_and_stream():
        try:
            logging.info("Reading and decoding image.")
            import io
            image = Image.open(io.BytesIO(content)).convert("RGB")
            img_np = np.array(image)
            logging.info("Image decoded successfully.")

            # Run YOLO detection
            logging.info("Starting YOLO detection.")
            det_results = yolo_model(img_np)
            box_infos = []
            
            img_with_boxes = img_np # Default to original image

            if det_results:
                # Calculate dynamic line width
                image_width = img_np.shape[1]
                line_width = max(1, int(image_width / 800))
                # Use the plot() method from the first result object, with custom line thickness
                img_with_boxes_bgr = det_results[0].plot(line_width=line_width)
                # Convert BGR to RGB
                img_with_boxes = img_with_boxes_bgr[:, :, ::-1]

                for r in det_results:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        if int(cls) == BOOK_CLASS_INDEX:
                            box_infos.append({
                                "box": box.tolist(),
                                "prob": float(conf)
                            })


            # Stream the initial image with all boxes
            import io
            initial_io = io.BytesIO()
            Image.fromarray(img_with_boxes).save(initial_io, format="PNG")
            initial_b64 = base64.b64encode(initial_io.getvalue()).decode("utf-8")
            yield json.dumps({"initial_image_with_boxes": initial_b64}) + "\n"
            await asyncio.sleep(0.1)

            logging.info(f"YOLO detection found {len(box_infos)} books.")
            
            if not box_infos:
                logging.warning("No books found in image.")
                yield json.dumps({"error": "No books found in image."}) + "\n"
                return

            for i, box_info in enumerate(box_infos):
                if i > ABORT_BOOK_COUNT:
                    logging.warning("aborting due to too many books")
                    break
                logging.info(f"Processing book {i+1}/{len(box_infos)}.")
                bbox = box_info["box"]
                prob = box_info["prob"]
                
                # Crop YOLO box image
                logging.info(f"Cropping YOLO box for book {i+1}.")
                x1, y1, x2, y2 = map(int, bbox)
                yolo_crop = img_np[y1:y2, x1:x2]
                import io
                yolo_io = io.BytesIO()
                Image.fromarray(yolo_crop).save(yolo_io, format="PNG")
                yolo_b64 = base64.b64encode(yolo_io.getvalue()).decode("utf-8")

                logging.info(f"Running SAM segmentation for book {i+1}.")
                sam_predictor.set_image(img_np)
                x1, y1, x2, y2 = map(int, bbox)
                input_box = np.array([x1, y1, x2, y2])
                masks, _, _ = sam_predictor.predict(
                    box=input_box[None, :],
                    multimask_output=False
                )
                mask_raw = masks[0]
                
                # Find bounding box from mask using numpy
                rows = np.any(mask_raw, axis=1)
                cols = np.any(mask_raw, axis=0)
                if not np.any(rows) or not np.any(cols):
                    # Handle empty mask
                    x, y, w, h = 0, 0, 0, 0
                else:
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin

                cropped_mask = mask_raw[y:y+h, x:x+w]
                cropped_image = img_np[y:y+h, x:x+w]
                segmented_image = np.zeros((h, w, 4), dtype=np.uint8)
                segmented_image[:, :, :3] = cropped_image
                segmented_image[:, :, 3] = cropped_mask * 255
                
                temp_path = f"temp_upload_{hash(tuple(bbox))}.png"
                logging.info(f"Saving segmented image to temporary file: {temp_path}")
                Image.fromarray(segmented_image).save(temp_path)
                
                seg_io = io.BytesIO()
                Image.fromarray(segmented_image).save(seg_io, format="PNG")
                seg_b64 = base64.b64encode(seg_io.getvalue()).decode("utf-8")

                # Perform OCR on the segmented image
                logging.info(f"Performing OCR for book {i+1}.")
                ocr_response = ocr(OLLAMA_HOST, MODEL, PROMPT, temp_path)
                logging.info(f"OCR for book {i+1} completed.")
                logging.info(f"OCR result: {ocr_response}")
                
                # Clean up the temporary file
                logging.info(f"Removing temporary file: {temp_path}")
                os.remove(temp_path)

                try:
                    # The OCR result is a string that needs to be parsed into a JSON object
                    logging.info("Parsing OCR response.")
                    ocr_result_json = ocr_response  # no json.loads needed
                except:
                    # If the OCR result is not a valid JSON, treat it as a plain string
                    logging.warning("OCR response is not valid JSON. Treating as plain text.")
                    ocr_result_json = {"text": ocr_response}


                result = {
                    "yolo_box": bbox,
                    "yolo_prob": prob,
                    "yolo_crop_b64": yolo_b64,
                    "sam_segmentation_b64": seg_b64,
                    "ocr_result": ocr_result_json
                }
                logging.info(f"Streaming result for book {i+1}.")
                yield json.dumps(result) + "\n"
                await asyncio.sleep(0.1)  # Small delay to allow stream to flush

        except Exception as e:
            logging.error(f"Error processing and streaming image: {e}", exc_info=True)
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(process_and_stream(), media_type="application/x-ndjson")
