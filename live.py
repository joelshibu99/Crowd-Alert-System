import streamlit as st
import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import pandas as pd
import datetime
import time

st.set_page_config(page_title="Eagle Eye", layout="wide")
st.title("Eagle Eye  Real-Time Crowd Alert System")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Zones
def get_zone(x, frame_width):
    if x < frame_width / 3:
        return "Left"
    elif x < 2 * frame_width / 3:
        return "Center"
    else:
        return "Right"

# Play alert sound in a separate thread
def play_alert():
    threading.Thread(target=playsound, args=("alert.wav",), daemon=True).start()

# Sidebar: zone limits
st.sidebar.header("Zone Limits")
limits = {
    "Left": st.sidebar.number_input("Left Zone Limit", min_value=1, value=5),
    "Center": st.sidebar.number_input("Center Zone Limit", min_value=1, value=5),
    "Right": st.sidebar.number_input("Right Zone Limit", min_value=1, value=5)
}

# Start / Stop
start_button = st.sidebar.button("Start Detection")
stop_button = st.sidebar.button("Stop Detection")

# Placeholders for video and chart
stframe = st.empty()
chart_placeholder = st.empty()

# Historical data for charts
df_history = pd.DataFrame(columns=["Left", "Center", "Right", "Total"])

# Overcrowding log
zone_alert_log = {z: [] for z in limits.keys()}
last_alert_time = 0  # cooldown for sound

# Main detection loop
if start_button:
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        st.error(" Could not open webcam.")
    else:
        st.info(" Webcam started. Press 'Stop Detection' to end.")
        while True:
            if stop_button:
                st.warning(" Detection stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                st.error(" Failed to capture frame.")
                break

            frame_height, frame_width = frame.shape[:2]

            # YOLO detection
            results = model(frame, verbose=False)
            detections = results[0].boxes

            zone_counts = {"Left": 0, "Center": 0, "Right": 0}
            total_persons = 0

            # Count people and draw minimal boxes
            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if model.names[cls].lower() == "person" and conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) // 2
                    zone = get_zone(x_center, frame_width)
                    zone_counts[zone] += 1
                    total_persons += 1

                    # Minimal green box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Zone dividers
            cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 2)
            cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 2)

            # Display total persons
            cv2.putText(frame, f"Total Persons: {total_persons}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Alerts and logging
            now = datetime.datetime.now()
            for i, (zone, count) in enumerate(zone_counts.items()):
                y_pos = 70 + (i * 30)
                color = (255, 255, 255)
                text = f"{zone}: {count}"

                if count > limits[zone]:
                    color = (0, 0, 255)
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    zone_alert_log[zone].append(timestamp)

                    # Play alert with 2-sec cooldown
                    if (now - datetime.datetime.fromtimestamp(last_alert_time)).total_seconds() > 2:
                        play_alert()
                        last_alert_time = now.timestamp()

                    text = f"{zone}: OVERCROWDED ({count})"

                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            # Update historical data
            df_history.loc[now] = [zone_counts["Left"], zone_counts["Center"], zone_counts["Right"], total_persons]
            chart_placeholder.line_chart(df_history)

            time.sleep(0.03)  # small delay to reduce CPU usage

        cap.release()

        # Display final log table
        st.subheader(" Overcrowding Log")
        log_data = []
        for zone, times in zone_alert_log.items():
            for t in times:
                log_data.append({"Zone": zone, "Time": t})
        df_log = pd.DataFrame(log_data)
        st.dataframe(df_log if not df_log.empty else "No overcrowding detected.")
