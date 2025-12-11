import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import datetime
import time

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")
alert_sound.set_volume(0.3)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define zones
def get_zone(x, frame_width):
    if x < frame_width / 3:
        return "Left"
    elif x < 2 * frame_width / 3:
        return "Center"
    else:
        return "Right"

# Get overcrowding limits with validation
print("Enter overcrowding limits for each zone :")
limits = {}
for zone in ["Left", "Center", "Right"]:
    while True:
        try:
            value = int(input(f"Limit for {zone} zone: "))
            if value <= 0:
                print(" Limit must be a positive integer. Try again.")
                continue
            limits[zone] = value
            break
        except ValueError:
            print(" Invalid input. Enter a valid integer.")

# Overcrowding timestamps
zone_alert_log = {z: [] for z in limits.keys()}
last_alert_time = 0  # for beep cooldown

# Open video
video_path = "video/sample_video.mp4"  # replace with your input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(" Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    zone_counts = {"Left": 0, "Center": 0, "Right": 0}
    total_persons = 0

    # Count people in each zone and draw minimal boxes
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[int(cls)].lower() == "person" and conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2

            zone = get_zone(x_center, frame_width)
            zone_counts[zone] += 1
            total_persons += 1

            # Minimal green box (no label clutter)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw zone dividers
    cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 2)
    cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 2)

    # Display total person count
    cv2.putText(frame, f"Total Persons: {total_persons}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display zone-wise counts and overcrowding alerts with timestamp
    for i, (zone, count) in enumerate(zone_counts.items()):
        y_pos = 70 + (i * 30)
        color = (255, 255, 255)

        if count > limits[zone]:
            color = (0, 0, 255)
            now = time.time()
            if now - last_alert_time > 2:  # cooldown
                alert_sound.play()
                last_alert_time = now

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            zone_alert_log[zone].append(timestamp)
            text = f"{zone}: OVERCROWDED ({count}) at {timestamp}"
        else:
            text = f"{zone}: {count}"

        cv2.putText(frame, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Crowd Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print overcrowding log
print("\n--- Overcrowding Log ---")
for zone, times in zone_alert_log.items():
    if times:
        print(f"{zone}:")
        for t in times:
            print(f"  - {t}")
    else:
        print(f"{zone}: No overcrowding detected.")
