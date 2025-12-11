import cv2
import streamlit as st
import numpy as np
import tempfile
import time
import pygame

# Initialize Pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def play_beep(frequency=440, duration=500, volume=0.5):
    sample_rate = 44100
    n_samples = int(sample_rate * duration / 1000)
    t = np.linspace(0, duration / 1000, n_samples, endpoint=False)
    wave = 32767 * volume * np.sin(2 * np.pi * frequency * t)
    wave = wave.astype(np.int16)
    wave = np.reshape(wave, (n_samples, 1))  # Mono audio
    sound = pygame.sndarray.make_sound(wave)
    sound.play()
    pygame.time.delay(duration)


def draw_zones(frame, num_zones):
    h, w, _ = frame.shape
    zone_height = h // num_zones
    zones = []
    for i in range(num_zones):
        top_left = (0, i * zone_height)
        bottom_right = (w, (i + 1) * zone_height)
        zones.append((top_left, bottom_right))
    return zones


def detect_people(frame, net, ln):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # class_id 0 is person
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    return boxes, idxs


def process_video(video_path, num_zones, zone_limits):
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    cap = cv2.VideoCapture(video_path)
    logs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        zones = draw_zones(frame, num_zones)
        boxes, idxs = detect_people(frame, net, ln)
        zone_counts = [0] * num_zones

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                cy = y + h // 2
                for z, (tl, br) in enumerate(zones):
                    if tl[1] <= cy < br[1]:
                        zone_counts[z] += 1

        for z, (tl, br) in enumerate(zones):
            color = (0, 255, 0)
            if zone_counts[z] > zone_limits[z]:
                color = (0, 0, 255)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logs.append(f"Zone {z+1} overcrowded at {timestamp}")
                play_beep()
            cv2.rectangle(frame, tl, br, color, 2)
            cv2.putText(frame, f"Zone {z+1}: {zone_counts[z]}/{zone_limits[z]}", (tl[0]+10, tl[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    return logs


# Streamlit interface
st.title("Crowd Density Alert System")
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

num_zones = st.slider("Select number of zones", 1, 5, 3)
zone_limits = []
for i in range(num_zones):
    limit = st.number_input(f"People limit for Zone {i+1}", min_value=1, value=5)
    zone_limits.append(limit)

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    logs = process_video(video_path, num_zones, zone_limits)

    if logs:
        st.subheader("Alert Log")
        for log in logs:
            st.write(log)
    else:
        st.success("No overcrowding detected.")