import numpy as np
from PIL import Image
import io
import base64
import logging

def process_yolo_detection(img_np, det_results):
    """Process YOLO detection results and return annotated image and box information."""
    box_infos = []
    img_with_boxes = img_np  # Default to original image

    if det_results:
        # Calculate dynamic line width
        image_width = img_np.shape[1]
        line_width = max(1, int(image_width / 800))
        
        # Use the plot() method from the first result object
        img_with_boxes_bgr = det_results[0].plot(line_width=line_width)
        img_with_boxes = img_with_boxes_bgr[:, :, ::-1]  # Convert BGR to RGB

        # Extract box information
        for r in det_results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                box_infos.append({
                    "box": box.tolist(),
                    "class": cls.item(),
                    "prob": conf.item()
                })

    return img_with_boxes, box_infos

def process_sam_segmentation(img_np, sam_predictor, bbox):
    """Process SAM segmentation for a given bounding box."""
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
        return None, None, None, None
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    cropped_mask = mask_raw[ymin:ymax, xmin:xmax]
    cropped_image = img_np[ymin:ymax, xmin:xmax]
    
    segmented_image = np.zeros((*cropped_image.shape[:2], 4), dtype=np.uint8)
    segmented_image[:, :, :3] = cropped_image
    segmented_image[:, :, 3] = cropped_mask * 255
    
    return segmented_image, xmin, ymin, cropped_mask

def encode_image_base64(image_array):
    """Convert numpy array image to base64 string."""
    img_io = io.BytesIO()
    Image.fromarray(image_array).save(img_io, format="PNG")
    return base64.b64encode(img_io.getvalue()).decode("utf-8")