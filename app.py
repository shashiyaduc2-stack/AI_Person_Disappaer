import streamlit as st
import av
import cv2
import numpy as np
import threading

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Person Disappear",
    page_icon="👤",
    layout="wide"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n-seg.pt")


model = load_model()


# =========================================================
# SHARED STATE
# =========================================================

lock = threading.Lock()


class SharedState:
    def __init__(self):
        self.background = None
        self.latest_frame = None
        self.background_ready = False
        self.disappear_mode = False


# One state object for this Streamlit session
if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()


state = st.session_state.shared_state


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class VideoProcessor:
    def __init__(self):
        self.background = None
        self.disappear_mode = False
        self.latest_frame = None

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Save latest frame
        with lock:
            self.latest_frame = image.copy()

        # Run YOLO segmentation
        results = model(
            image,
            verbose=False,
            imgsz=320
        )

        result = results[0]

        final_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        if result.masks is not None:

            for i, mask_data in enumerate(result.masks.data):

                class_id = int(result.boxes.cls[i])

                # COCO class 0 = person
                if class_id != 0:
                    continue

                mask = mask_data.cpu().numpy()

                mask = cv2.resize(
                    mask,
                    (
                        image.shape[1],
                        image.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = (
                    mask > 0.5
                ).astype(np.uint8) * 255

                final_mask = cv2.bitwise_or(
                    final_mask,
                    mask
                )

        # Slightly expand mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )

        final_mask = cv2.dilate(
            final_mask,
            kernel,
            iterations=1
        )

        # Person disappearance
        if (
            self.disappear_mode
            and self.background is not None
        ):

            background = cv2.resize(
                self.background,
                (
                    image.shape[1],
                    image.shape[0]
                )
            )

            # Smooth edge
            smooth_mask = cv2.GaussianBlur(
                final_mask,
                (21, 21),
                0
            )

            alpha = (
                smooth_mask.astype(np.float32)
                / 255.0
            )

            alpha = alpha[:, :, np.newaxis]

            output = (
                background * alpha
                + image * (1 - alpha)
            )

            output = np.clip(
                output,
                0,
                255
            ).astype(np.uint8)

        else:

            output = image

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


# =========================================================
# HEADER
# =========================================================

st.title("👤 AI Person Disappear")
st.caption(
    "Real-Time Computer Vision • YOLO Segmentation • OpenCV"
)


# =========================================================
# START LIVE CAMERA
# =========================================================

st.write("### 🎥 Live Camera")

ctx = webrtc_streamer(
    key="person-disappear",
    video_processor_factory=VideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# =========================================================
# CONTROLS
# =========================================================

st.write("### 🎛️ Controls")


col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# CAPTURE BACKGROUND
# ---------------------------------------------------------

with col1:

    if st.button(
        "📸 Capture Background",
        use_container_width=True
    ):

        if (
            ctx.video_processor
            and ctx.video_processor.latest_frame
            is not None
        ):

            with lock:

                state.background = (
                    ctx.video_processor
                    .latest_frame
                    .copy()
                )

                state.background_ready = True

                ctx.video_processor.background = (
                    state.background.copy()
                )

            st.success(
                "Background captured! ✅"
            )

        else:

            st.warning(
                "Start the camera first."
            )


# ---------------------------------------------------------
# TOGGLE DISAPPEAR
# ---------------------------------------------------------

with col2:

    if st.button(
        "✨ Toggle Disappear",
        use_container_width=True
    ):

        if not state.background_ready:

            st.warning(
                "Capture the background first."
            )

        elif ctx.video_processor:

            state.disappear_mode = (
                not state.disappear_mode
            )

            ctx.video_processor.disappear_mode = (
                state.disappear_mode
            )


# ---------------------------------------------------------
# RESET
# ---------------------------------------------------------

with col3:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):

        state.background = None
        state.background_ready = False
        state.disappear_mode = False

        if ctx.video_processor:

            ctx.video_processor.background = None
            ctx.video_processor.disappear_mode = False

        st.rerun()


# =========================================================
# STATUS
# =========================================================

st.divider()

col1, col2 = st.columns(2)


with col1:

    if state.background_ready:

        st.success(
            "🟢 BACKGROUND READY"
        )

    else:

        st.warning(
            "🟡 BACKGROUND NOT CAPTURED"
        )


with col2:

    if state.disappear_mode:

        st.success(
            "✨ DISAPPEARANCE: ON"
        )

    else:

        st.info(
            "DISAPPEARANCE: OFF"
        )


# =========================================================
# INSTRUCTIONS
# =========================================================

st.write("### 📋 How to use")

st.markdown(
    """
    **1.** Click **START** above and allow camera access.

    **2.** Move yourself out of the camera view.

    **3.** Click **Capture Background**.

    **4.** Stand in front of the camera.

    **5.** Click **Toggle Disappear**.

    **6.** The detected person will be replaced
    with the captured background.
    """
)