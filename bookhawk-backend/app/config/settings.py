import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.178.39:11434")
MODEL = os.getenv("MODEL", "qwen2.5vl:7b")
PROMPT = os.getenv("PROMPT", "You will receive a image of a book front view in a book shelf. Extract the author and title of the book image. Return the output as JSON. Output the JSON as {'author': .. , 'title': .., 'language': ..}. Do not mistake the genre or publisher as the author. If you cannot extract the author from the image, output the author that has published the book that you know of.")

# Model Paths
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov10x.pt")
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "sam_vit_b_01ec64.pth")

# Model Configuration
BOOK_CLASS_INDEX = int(os.getenv("BOOK_CLASS_INDEX", "73"))
ABORT_BOOK_COUNT = int(os.getenv("ABORT_BOOK_COUNT", "3"))