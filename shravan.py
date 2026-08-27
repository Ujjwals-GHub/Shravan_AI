import os
import av
import cv2
import time
import tempfile
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
    model_path = "best.pt" if os.path.exists("best.pt") else "yolo11n.pt"
    return YOLO(model_path)

model = load_yolo_model()

# ==========================================
# 2. SETTINGS & CONSTANTS
# ==========================================
CONF_THRESHOLD = 0.50
DANGEROUS_OBJECTS = {"bed", "chair", "table", "laptop", "window", "door", "person"}

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

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # Frame skipping optimization: process inference every 3rd frame
        if self.frame_counter % 3 == 0:
            annotated_img, _ = process_detection(img)
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. USER INTERFACE MODES
# ==========================================
mode = st.radio("Select Navigation Mode:", ["📸 Snapshot & Voice Alert", "🎥 Live WebRTC Stream"], horizontal=True)

if mode == "🎥 Live WebRTC Stream":
    st.info("Click **START** to initialize the live camera feed with directional bounding boxes.")
    webrtc_streamer(
        key="shravan-feed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif mode == "📸 Snapshot & Voice Alert":
    camera_img = st.camera_input("Capture live view:")
    
    if camera_img is not None:
        # Convert file buffer to OpenCV BGR image
        bytes_data = camera_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        annotated_frame, detected_threats = process_detection(cv2_img)
        rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        st.image(rgb_preview, caption="Detection Analysis", use_container_width=True)

        if detected_threats:
            warning_text = ", ".join(detected_threats)
            st.error(f"⚠️ **Obstacles Detected:** {warning_text}")
            
            # Generate cross-platform voice feedback
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
