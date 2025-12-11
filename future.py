import streamlit as st
import cv2
from ultralytics import YOLO
from playsound import playsound
import tempfile
import threading
import pandas as pd
import time
import datetime
import os
import re
import json
import google.generativeai as genai

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Eagle Eye", layout="wide")
st.title("Eagle Eye Crowd Alert System")

# -------------------- Google Gemini Setup --------------------
API_KEY = st.secrets.get("GEMINI_API_KEY") 
if not API_KEY:
    st.error(" Gemini API Key not found! Set it in Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# -------------------- Load YOLOv8 --------------------
model = YOLO("yolov8n.pt")

# -------------------- Helper Functions --------------------
def get_zone(x, frame_width):
    if x < frame_width / 3:
        return "Left"
    elif x < 2 * frame_width / 3:
        return "Center"
    else:
        return "Right"

def play_alert():
    threading.Thread(target=playsound, args=("alert.wav",), daemon=True).start()

def generate_summary(zone_counts, total_persons):
    prompt = f"""
    Generate a short, human-readable report for event safety:

    Current crowd situation:
    - Left zone: {zone_counts['Left']} persons
    - Center zone: {zone_counts['Center']} persons
    - Right zone: {zone_counts['Right']} persons
    - Total: {total_persons} persons

    Mention which zones are overcrowded and suggest immediate action.
    """
    try:
        gen_model = genai.GenerativeModel(MODEL_NAME)
        response = gen_model.generate_content(prompt)
        summary = response.text.strip()
        # Remove code blocks if present
        summary = re.sub(r"```.*?```", "", summary, flags=re.DOTALL).strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# -------------------- Video Upload --------------------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

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

    # Placeholders
    stframe = st.empty()
    chart_placeholder = st.empty()
    summary_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(video_path)
        zone_alert_log = {z: [] for z in limits.keys()}
        last_alert_time = 0
        last_summary_time = 0
        summary_cooldown = 10  # seconds between Gemini API calls

        df_history = pd.DataFrame(columns=["Left", "Center", "Right", "Total"])

        while cap.isOpened():
            if stop_button:
                st.warning("Detection stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                st.info("Video finished.")
                break

            frame_height, frame_width = frame.shape[:2]
            results = model(frame, verbose=False)
            detections = results[0].boxes

            zone_counts = {"Left": 0, "Center": 0, "Right": 0}
            total_persons = 0

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

            # Total persons
            cv2.putText(frame, f"Total Persons: {total_persons}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Alerts & logging
            now = datetime.datetime.now()
            for i, (zone, count) in enumerate(zone_counts.items()):
                y_pos = 70 + (i * 30)
                color = (255, 255, 255)
                text = f"{zone}: {count}"

                if count > limits[zone]:
                    color = (0, 0, 255)
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    zone_alert_log[zone].append(timestamp)
                    # Cooldown for sound
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

            # Generate summary with cooldown
            if (now - datetime.datetime.fromtimestamp(last_summary_time)).total_seconds() > summary_cooldown:
                summary_text = generate_summary(zone_counts, total_persons)
                summary_placeholder.subheader("Crowd Summary (Gemini AI)")
                summary_placeholder.write(summary_text)
                last_summary_time = now.timestamp()

            time.sleep(0.05)

        cap.release()

        # Final log table
        st.subheader("Overcrowding Log")
        log_data = []
        for zone, times in zone_alert_log.items():
            for t in times:
                log_data.append({"Zone": zone, "Time": t})
        df_log = pd.DataFrame(log_data)
        st.dataframe(df_log if not df_log.empty else "No overcrowding detected.")
