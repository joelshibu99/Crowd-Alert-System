import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt' for better accuracy

# Load the video
cap = cv2.VideoCapture('Times Square Crowd People 2 HD Video Background(1080P_HD).mp4')  # Change to your video path
threshold = 10  # Set your own limit

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)[0]

    # Extract bounding boxes for people only (class ID 0 = person in COCO dataset)
    person_boxes = [box for box in results.boxes.data if int(box[5]) == 0]
    count = len(person_boxes)

    # Draw boxes and labels
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add overlay text
    cv2.putText(frame, f"People: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if count > threshold:
        cv2.putText(frame, "⚠️ Overcrowding Alert!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the frame
    cv2.imshow('Crowd Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
