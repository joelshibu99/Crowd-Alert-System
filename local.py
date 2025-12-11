import cv2
from ultralytics import YOLO
from playsound import playsound
import threading
import pandas as pd
import time
import datetime
import re
import google.generativeai as genai
import os

# -------------------- CONFIGURATION --------------------
VIDEO_PATH = "video\sample_video.mp4"  # Change to your video path
ALERT_SOUND = "alert.wav"  # Sound to play
MODEL_PATH = "yolov8n.pt"
API_KEY = os.getenv("GEMINI_API_KEY")  # Or directly assign your key here (not recommended)

# Zone person limits
ZONE_LIMITS = {"Left": 5, "Center": 5, "Right": 5}

if not API_KEY:
    print("❌ Gemini API Key not found! Set it as environment variable GEMINI_API_KEY.")
    exit()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"
model = YOLO(MODEL_PATH)

# -------------------- HELPER FUNCTIONS --------------------
def get_zone(x, frame_width):
    if x < frame_width / 3:
        return "Left"
    elif x < 2 * frame_width / 3:
        return "Center"
    else:
        return "Right"

def play_alert():
    threading.Thread(target=playsound, args=(ALERT_SOUND,), daemon=True).start()

def generate_summary(zone_counts, total_persons):
    prompt = f"""
    Generate a short, human-readable safety summary.

    Current crowd situation:
    - Left zone: {zone_counts['Left']} persons
    - Center zone: {zone_counts['Center']} persons
    - Right zone: {zone_counts['Right']} persons
    - Total: {total_persons} persons

    Mention which zones are overcrowded and suggest actions.
    """
    try:
        gen_model = genai.GenerativeModel(MODEL_NAME)
        response = gen_model.generate_content(prompt)
        summary = response.text.strip()
        summary = re.sub(r"```.*?```", "", summary, flags=re.DOTALL).strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# -------------------- MAIN LOGIC --------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        return

    zone_alert_log = {z: [] for z in ZONE_LIMITS.keys()}
    df_history = pd.DataFrame(columns=["Left", "Center", "Right", "Total"])

    last_alert_time = 0
    last_summary_time = 0
    summary_cooldown = 10  # seconds between Gemini calls

    print("✅ Detection started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Draw zone dividers
        cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 2)
        cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_persons}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        now = datetime.datetime.now()

        # Zone text and alerts
        for i, (zone, count) in enumerate(zone_counts.items()):
            y_pos = 70 + (i * 30)
            color = (255, 255, 255)
            text = f"{zone}: {count}"
            if count > ZONE_LIMITS[zone]:
                color = (0, 0, 255)
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                zone_alert_log[zone].append(timestamp)
                if (now - datetime.datetime.fromtimestamp(last_alert_time)).total_seconds() > 2:
                    play_alert()
                    last_alert_time = now.timestamp()
                text = f"{zone}: OVERCROWDED ({count})"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame
        cv2.imshow("Eagle Eye - Crowd Alert", frame)

        # Save history
        df_history.loc[now] = [zone_counts["Left"], zone_counts["Center"], zone_counts["Right"], total_persons]

        # Periodic Gemini summary
        if (now - datetime.datetime.fromtimestamp(last_summary_time)).total_seconds() > summary_cooldown:
            summary_text = generate_summary(zone_counts, total_persons)
            print("\n------ Crowd Summary (Gemini AI) ------")
            print(summary_text)
            print("---------------------------------------\n")
            last_summary_time = now.timestamp()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Detection stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final log
    print("\n===== Overcrowding Log =====")
    for zone, times in zone_alert_log.items():
        if times:
            print(f"{zone} zone:")
            for t in times:
                print(f"  - {t}")
        else:
            print(f"{zone} zone: No overcrowding detected.")
    print("============================")

    # Save log to CSV
    df_log = []
    for z, times in zone_alert_log.items():
        for t in times:
            df_log.append({"Zone": z, "Time": t})
    if df_log:
        log_df = pd.DataFrame(df_log)
        log_df.to_csv("overcrowding_log.csv", index=False)
        print("\n📄 Log saved to 'overcrowding_log.csv'")
    else:
        print("\n✅ No overcrowding events recorded.")

if __name__ == "__main__":
    main()
