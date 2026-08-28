import os
import av
import cv2
import time
import json
import html
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
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
REPEAT_DELAY = 2.0  # seconds before the SAME obstacle set is announced again

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

_HIDE_IFRAME_JS = """
try {
    var frames = window.parent.document.querySelectorAll('iframe');
    var me = frames[frames.length - 1];
    if (me) {
        me.style.height = '0px';
        if (me.parentNode) {
            me.parentNode.style.height = '0px';
            me.parentNode.style.marginBottom = '0px';
        }
    }
} catch (e) {}
"""

_VOICE_ENGINE_JS = """
<script>
(function() {
    __HIDE_IFRAME__
    var lastNonce = null;
    var synth = window.speechSynthesis;

    function poll() {
        var marker = window.parent.document.getElementById('tts-marker');
        if (marker) {
            var nonce = marker.getAttribute('data-nonce');
            var text = marker.textContent || "";
            if (nonce && nonce !== lastNonce && text.trim().length > 0 && synth) {
                lastNonce = nonce;
                synth.cancel();  // don't let alerts pile up in a queue
                var utter = new SpeechSynthesisUtterance(text);
                utter.lang = "en-US";
                utter.rate = 1.05;
                synth.speak(utter);
            }
        }
        setTimeout(poll, 250);
    }
    poll();
})();
</script>
"""

_SPEAK_ONCE_JS = """
<script>
(function() {
    __HIDE_IFRAME__
    var synth = window.speechSynthesis;
    if (synth) {
        synth.cancel();
        var utter = new SpeechSynthesisUtterance(__TEXT__);
        utter.lang = "en-US";
        synth.speak(utter);
    }
})();
</script>
"""

def inject_voice_engine():
    """Call ONCE per session, right when the stream starts, to arm the
    persistent speech engine described above."""
    components.html(_VOICE_ENGINE_JS.replace("__HIDE_IFRAME__", _HIDE_IFRAME_JS), height=0)

def set_voice_alert(placeholder, text: str):
    """Writes new alert text into the hidden mailbox the voice engine polls."""
    nonce = str(time.time())
    placeholder.markdown(
        f'<div id="tts-marker" data-nonce="{nonce}" style="display:none">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )

def speak_once(text: str):
    """One-shot speech for a fresh gesture-triggered event (e.g. a new
    camera snapshot). Safe here because each new snapshot is itself a
    fresh rerun tied to the user's capture action."""
    js = _SPEAK_ONCE_JS.replace("__HIDE_IFRAME__", _HIDE_IFRAME_JS).replace("__TEXT__", json.dumps(text))
    components.html(js, height=0)

# ==========================================
# 4. CORE DETECTION & SPATIAL LOGIC
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
# 5. WEBRTC VIDEO STREAMING PROCESSOR
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
# 6. USER INTERFACE MODES
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
        # Arm the persistent speech engine ONCE per play session - this runs
        # as part of the rerun triggered by clicking START, so the browser
        # associates the engine with that click and keeps allowing it to
        # speak afterwards.
        inject_voice_engine()

        status_text = st.empty()
        voice_marker = st.empty()

        last_message = ""
        last_spoken_time = 0.0

        while True:
            if ctx.video_processor:
                current_threats = ctx.video_processor.detected_threats

                if current_threats:
                    unique_msg = ", ".join(current_threats)
                    status_text.error(f"⚠️ **Obstacles Detected:** {unique_msg}")

                    now = time.time()

                    if (unique_msg != last_message) or (now - last_spoken_time > REPEAT_DELAY):
                        set_voice_alert(voice_marker, unique_msg)
                        last_message = unique_msg
                        last_spoken_time = now
                else:
                    status_text.success("✅ **Path Clear:** No immediate obstacles detected.")
                    # Do NOT clear last_message here. Keeping it prevents spam if the camera flickers.

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
            speak_once(warning_text)
        else:
            st.success("✅ **Path Clear:** No immediate obstacles detected.")
            speak_once("Path clear. No obstacles ahead.")
