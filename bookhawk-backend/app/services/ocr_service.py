import os
import base64
import logging
import requests
from app.config.settings import OLLAMA_HOST, MODEL, PROMPT

def ocr_from_file(image_path, ollama_host=OLLAMA_HOST, model_name=MODEL, prompt=PROMPT):
    """Perform OCR on an image file using Ollama API."""
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
        ocr_result = response.json()['message']['content']
        logging.info(f"Ollama API response: {ocr_result}")
        
        return ocr_result
        
    except Exception as e:
        logging.error(f"Error in OCR function: {e}")
        return {"error": str(e)}