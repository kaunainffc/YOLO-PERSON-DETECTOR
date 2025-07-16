# Streamlit App: Live Person Detection with YOLOv8 and Webcam

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Live Person Detection", layout="centered")
st.title("🧍 Real-time Person Detection using YOLOv8")

st.markdown("""
Upload a snapshot from your webcam or use a frame, and this app will detect if a **person** is present using a YOLOv8 model.
If a person is detected, a beep will be played and the frame will be saved.
""")

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload or capture a frame
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    # Run detection
    result = model.predict(source=frame, show=False, conf=0.5)
    results = result[0]
    detected_frame = results.plot()

    # Check for person detection (class 0 in COCO)
    classes = results.boxes.cls.cpu().numpy().astype(int)
    if 0 in classes:
        st.success("✅ Person detected!")

        # Save image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            cv2.imwrite(f.name, frame[:, :, ::-1])
            st.image(detected_frame, caption="Detection Result", use_column_width=True)
            st.download_button("Download Detected Image", open(f.name, "rb"), file_name="person_detected.jpg")
    else:
        st.warning("No person detected.")
        st.image(detected_frame, caption="Detection Result", use_column_width=True)

st.markdown("---")
st.caption("Powered by YOLOv8 and Streamlit")
