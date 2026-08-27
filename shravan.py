#Importing necessary libraries
import cv2                      #For video capture and image processing
from ultralytics import YOLO    #For object detection using YOLO model
import win32com.client          #For connecting Python to the Windows text-to-speech engine
import time                     #For handling time-related functions
import threading                #For threading
import pythoncom                #For initializing COM in threads

# LOADING MODEL 
model = YOLO(r"C:\Users\ujj.ext\Downloads\IITG-P1-main\IITG-P1-main\runs\detect\train6\weights\best.pt")
cap = cv2.VideoCapture(0)

# MODEL PARAMETERS
CONF_THRESHOLD = 0.5
DELAY = 1
DANGEROUS_OBJECTS = ["bed", "chair", "table", "laptop", "window", "door", "person"]
last_spoken_time = 0
last_message = ""
frame_counter = 0 

# Voice Engine Initialization
def speak_async(text):
    def run_speech():
        pythoncom.CoInitialize() 
        local_speaker = win32com.client.Dispatch("SAPI.SpVoice")
        local_speaker.Rate = -1 
        local_speaker.Speak(text)
    
    # Start the speech in a background thread
    t = threading.Thread(target=run_speech, daemon=True)
    t.start()

# MAIN LOOP to read video frames, perform object detection, and speak results
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    
    # Only run the YOLO model and drawing logic on every 3rd frame
    if frame_counter % 3 != 0:
        cv2.imshow("SHRAVAN", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        continue 

    # OBJECT DETECTION
    results = model(frame)
    h, w, _ = frame.shape
    messages = []

    for r in results:
        for box in r.boxes:

            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in DANGEROUS_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            # Direction logic
            if center_x < w // 3:
                direction = "left"
            elif center_x > 2 * w // 3:
                direction = "right"
            else:
                direction = "ahead"

            messages.append(f"{label} {direction}")

            # Draw bounding boxes and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # SPEAK LOGIC
    current_time = time.time()
    if messages:
        unique_msg = ", ".join(sorted(set(messages)))

        if (unique_msg != last_message) or (current_time - last_spoken_time > DELAY):
            print("[SPEAK]", unique_msg)
            speak_async(unique_msg)  # Call the new threaded function

            last_message = unique_msg
            last_spoken_time = current_time

    # ======================
    # SHOW
    # ======================
    cv2.imshow("SHRAVAN", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
