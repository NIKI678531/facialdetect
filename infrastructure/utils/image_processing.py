import logging
import os

import torch
from ultralytics import YOLO

device = 'mps'


# def load_yolo_model():
#     model = YOLO("/Users/dingpengxu1/PycharmProjects/detectCard/yolov8n.pt").to(device)
#     return model
model = YOLO(model="yolov8n.pt").to(device)

# Function to calculate the area of a bounding box
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


# Function to process each image
def process_image(image):
    logging.info("Processing image")
    # Run inference on the image

    # results = load_yolo_model()(image, device=device)
    results = model(image)

    max_person_area = 0
    max_person_bbox = None

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 is 'person'
                bbox = box.xyxy[0].tolist()[:4]
                area = bbox_area(bbox)
                if area > max_person_area:
                    max_person_area = area
                    max_person_bbox = bbox

    if max_person_bbox:
        logging.info(f"Found person with bounding box: {max_person_bbox}")
        # Crop the image to the bounding box of the largest person
        x1, y1, x2, y2 = map(int, max_person_bbox)
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    logging.info("No person found in the image")
    return None
