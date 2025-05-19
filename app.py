import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# Load the YOLO model
model = YOLO("detect.pt")  # Replace with your custom model path

# Function to process image (for both uploaded images and webcam frames)
def process_image(image):
    if isinstance(image, str):  # If image is a file path
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):  # If image is a numpy array
        image = Image.fromarray(image).convert("RGB")
    
    img_np = np.array(image)
    results = model(img_np)
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img

# Function for image upload mode
def detect_image(image):
    if image is None:
        return None, "Please upload an image."
    result = process_image(image)
    return result, "Detection complete!"

# Function for webcam mode
def detect_webcam(video):
    if video is None:
        return None, "Please provide a video input."
    # For Gradio, video input is a file path
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "Error reading video frame."
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = process_image(frame_rgb)
    cap.release()
    return result, "Webcam detection complete!"

# Gradio interface
with gr.Blocks(title="YOLO Model Detection") as demo:
    gr.Markdown("# 📸 YOLO Model Detection")
    
    # Mode selection
    mode = gr.Radio(choices=["Image Upload", "Webcam Detection"], label="Choose Mode", value="Image Upload")
    
    # Image Upload Interface
    with gr.Group(visible=True) as image_group:
        gr.Markdown("### 🖼️ Upload an Image for Detection")
        image_input = gr.Image(type="filepath", label="Upload Image")
        image_output = gr.Image(label="Detection Result")
        image_status = gr.Textbox(label="Status")
        image_button = gr.Button("Detect Image")
    
    # Webcam Interface
    with gr.Group(visible=False) as webcam_group:
        gr.Markdown("### 🎥 Webcam Detection")
        gr.Markdown("Note: Upload a short video or use your webcam to capture a frame.")
        webcam_input = gr.Video(label="Webcam Input")
        webcam_output = gr.Image(label="Detection Result")
        webcam_status = gr.Textbox(label="Status")
        webcam_button = gr.Button("Detect Webcam")

    # Logic to toggle visibility based on mode
    def toggle_mode(selected_mode):
        return {
            image_group: gr.update(visible=selected_mode == "Image Upload"),
            webcam_group: gr.update(visible=selected_mode == "Webcam Detection")
        }

    mode.change(fn=toggle_mode, inputs=mode, outputs=[image_group, webcam_group])

    # Connect buttons to functions
    image_button.click(fn=detect_image, inputs=image_input, outputs=[image_output, image_status])
    webcam_button.click(fn=detect_webcam, inputs=webcam_input, outputs=[webcam_output, webcam_status])

# Launch the app
if __name__ == "__main__":
    demo.launch()