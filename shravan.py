import os
import av
import cv2
import time
import base64
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. PAGE CONFIGURATION & CACHED MODEL
# ==========================================
st.set_page_config(
    page_title="Shravan — Blind Assistant",
    page_icon="🦯",
    layout="centered"
)

st.title("🦯 Shravan — Assistive Vision System")
st.markdown("Real-time object detection and directional guidance for indoor navigation.")

@st.cache_resource
def load_yolo_model():
    # Looks for your custom best.pt, falls back to nano if missing
    model_path = "best.pt" if os.path.exists("best.pt") else "yolo11n.pt"
    return YOLO(model_path)

model = load_yolo_model()

# ==========================================
# 2. SETTINGS & CONSTANTS
# ==========================================
CONF_THRESHOLD = 0.50
DANGEROUS_OBJECTS = {"bed", "chair", "table", "laptop", "window", "door", "person"}

# Free Google STUN server for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 3. CORE DETECTION & SPATIAL LOGIC
# ==========================================
def process_detection(frame):
    """Processes a single BGR frame, draws annotations, and returns warnings."""
    h, w, _ = frame.shape
    results = model(frame, conf=CONF_THRESHOLD)
    messages = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in DANGEROUS_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2

            # Spatial awareness: 3 vertical thirds
            if center_x < w // 3:
                direction = "left"
            elif center_x > 2 * w // 3:
                direction = "right"
            else:
                direction = "ahead"

            messages.append(f"{label} {direction}")

            # Draw visual bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({direction})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return frame, sorted(list(set(messages)))

# ==========================================
# 4. WEBRTC VIDEO STREAMING PROCESSOR
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.frame_counter = 0
        self.detected_threats = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Process inference every 3rd frame to reduce CPU load
        if self.frame_counter % 3 == 0:
            annotated_img, threats = process_detection(img)
            self.detected_threats = threats 
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. USER INTERFACE MODES
# ==========================================
mode = st.radio("Select Navigation Mode:", ["📸 Snapshot & Voice Alert", "🎥 Live WebRTC Stream"], horizontal=True)

if mode == "🎥 Live WebRTC Stream":
    st.info("Click **START** to initialize the live camera feed with directional bounding boxes.")
    
    ctx = webrtc_streamer(
        key="shravan-feed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # ----------------------------------------------------
    # POLLING LOOP FOR LIVE AUDIO FEEDBACK
    # ----------------------------------------------------
    if ctx.state.playing:
        status_text = st.empty()
        audio_placeholder = st.empty()
        
        last_message = ""
        last_spoken_time = 0
        tts_cache = {}  
        DELAY = 2.0     
        
        while True:
            if ctx.video_processor:
                current_threats = ctx.video_processor.detected_threats
                
                if current_threats:
                    unique_msg = ", ".join(current_threats)
                    status_text.error(f"⚠️ **Obstacles Detected:** {unique_msg}")
                    
                    current_time = time.time()
                    
                    if (unique_msg != last_message) or (current_time - last_spoken_time > DELAY):
                        if unique_msg not in tts_cache:
                            tts = gTTS(text=unique_msg, lang="en")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                with open(fp.name, "rb") as f:
                                    tts_cache[unique_msg] = base64.b64encode(f.read()).decode()
                        
                        b64 = tts_cache[unique_msg]
                        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                        
                        last_message = unique_msg
                        last_spoken_time = current_time
                else:
                    status_text.success("✅ **Path Clear:** No immediate obstacles detected.")
                    last_message = ""
                    
            time.sleep(0.5) 

elif mode == "📸 Snapshot & Voice Alert":
    camera_img = st.camera_input("Capture live view:")
    
    if camera_img is not None:
        bytes_data = camera_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        annotated_frame, detected_threats = process_detection(cv2_img)
        rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        st.image(rgb_preview, caption="Detection Analysis", use_container_width=True)

        if detected_threats:
            warning_text = ", ".join(detected_threats)
            st.error(f"⚠️ **Obstacles Detected:** {warning_text}")
            
            tts = gTTS(text=warning_text, lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        else:
            st.success("✅ **Path Clear:** No immediate obstacles detected.")
            tts = gTTS(text="Path clear. No obstacles ahead.", lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
