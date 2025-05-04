import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

# Load your YOLO model
model = YOLO("detect.pt")  # Replace with your custom model path

# App title
st.title("📸 YOLO Model Detection")

# Sidebar for mode
mode = st.sidebar.selectbox("Choose Mode", ["Webcam Detection", "Image Upload"])

# ======================
# Webcam Mode
# ======================

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(annotated, format="rgb24")

if mode == "Webcam Detection":
    st.subheader("🎥 Webcam Detection")
    webrtc_streamer(key="yolo-webcam", video_transformer_factory=VideoTransformer)

# ======================
# Image Upload Mode
# ======================

elif mode == "Image Upload":
    st.subheader("🖼️ Upload an Image for Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        results = model(img_np)
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Detection Result", use_container_width=True)
        st.success("Detection complete!")
