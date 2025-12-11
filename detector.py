from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8s model
model = YOLO("yolov8n.pt")  # 'n' is for nano (small & fast)

def count_people_in_frame(frame):
    results = model(frame)
    detections = results[0].boxes

    person_count = 0
    for det in detections:
        cls_id = int(det.cls[0])
        if cls_id == 0:  # class 0 = person in COCO
            person_count += 1

    return person_count
