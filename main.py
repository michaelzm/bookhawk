import os
import base64
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import requests
from ultralytics import YOLO, SAM

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.178.39:11434")
MODEL = os.getenv("MODEL", "qwen2.5vl:7b")
PROMPT = os.getenv("PROMPT", "You will receive a image of a book front view in a book shelf. Extract the author and title of the book image. Return the output as JSON. Output the JSON as {'author': .. , 'title': .., 'language': ..}. Do not mistake the genre or publisher as the author. If you cannot extract the author from the image, output the author that has published the book that you know of.")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov10x.pt")
BOOK_CLASS_INDEX = int(os.getenv("BOOK_CLASS_INDEX", "73"))
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "sam2.1_b.pt")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

yolo_model = YOLO(YOLO_MODEL_PATH)
sam_model = SAM(SAM_MODEL_PATH)

def ocr(ollama_host, model_name, prompt, image_path):
    print("connect to ollama host:", ollama_host)
    try:
        if not os.path.exists(image_path):
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
        response = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()

@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        img_np = np.array(image)
        # Run YOLO detection
        det_results = yolo_model(img_np)
        box_infos = []
        for r in det_results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if int(cls) == BOOK_CLASS_INDEX:
                    box_infos.append({
                        "box": box.tolist(),
                        "prob": float(conf)
                    })
        if not box_infos:
            return JSONResponse({"error": "No books found in image."}, status_code=400)
        # Use first book box for demo
        bbox = box_infos[0]["box"]
        prob = box_infos[0]["prob"]
        # Crop YOLO box image
        x1, y1, x2, y2 = map(int, bbox)
        yolo_crop = img_np[y1:y2, x1:x2]
        import io
        yolo_io = io.BytesIO()
        Image.fromarray(yolo_crop).save(yolo_io, format="PNG")
        yolo_b64 = base64.b64encode(yolo_io.getvalue()).decode("utf-8")

        results = sam_model(img_np, bboxes=[bbox])
        masks = results[0].masks.data.cpu().numpy()
        mask_raw = masks[0]
        x, y, w, h = cv2.boundingRect(mask_raw.astype(np.uint8))
        cropped_mask = mask_raw[y:y+h, x:x+w]
        cropped_image = img_np[y:y+h, x:x+w]
        segmented_image = np.zeros((h, w, 4), dtype=np.uint8)
        segmented_image[:, :, :3] = cropped_image
        segmented_image[:, :, 3] = cropped_mask * 255
        temp_path = "temp_upload.png"
        Image.fromarray(segmented_image).save(temp_path)
        seg_io = io.BytesIO()
        Image.fromarray(segmented_image).save(seg_io, format="PNG")
        seg_b64 = base64.b64encode(seg_io.getvalue()).decode("utf-8")

        ocr_result = ocr(OLLAMA_HOST, MODEL, PROMPT, temp_path)
        os.remove(temp_path)
        return JSONResponse({
            "ocr_result": ocr_result,
            "yolo_box_image": yolo_b64,
            "yolo_box_prob": prob,
            "segmented_image": seg_b64
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
