from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
from PIL import Image
import os
from streamlit_cropper import st_cropper
import time

st.set_page_config(
    page_title="Dental Image Segmentation and Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dental Image Detection and Prediction Using YOLOv8 and SAM")

st.sidebar.header("DL Model Configuration")

model_type = st.sidebar.radio(
    "Choose Task", ['Detection', 'Segmentation']
)

# Default confidence value is 0.30
confidence = float(st.sidebar.slider(
    "Confidence Threshold (In Percentage %)",
    min_value=25,
    max_value=100,
    value=30
)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model for the following path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")

source_radio = st.sidebar.radio(
    "Select Your Source", settings.SOURCES_LIST
)

source_img = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image.", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                # Placeholder for the progress bar
                progress_bar = st.progress(0)

                if uploaded_image is not None:

                    # Upload progress
                    for percent_complete in range(0, 101, 10):
                        time.sleep(0.1)  # Simulate upload delay
                        progress_bar.progress(percent_complete)
                    progress_bar.empty()
                st.image(source_img, caption="Uploaded Image Successfully")
                
        except Exception as ex:
            st.error(f"Error loading image: {ex}")

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path
            )
            st.image(default_detected_image_path,
                     caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Disease and Stuff'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)

                try:
                    with st.expander("Detectioin Results"):
                        for box in boxes:
                            st.write(box.data)
                            class_label = model.names[int(box.cls)]
                            st.write(class_label)
                except Exception as ex:
                    st.write("No image has been uploaded.")

    with st.container():
        # Check`custom-labels` folder exists
        os.makedirs("custom-labels", exist_ok=True)

        # Save bounding coordinates
        def save_img_label_yolo_format(filename, img_width, img_height, bbox, label, cropped_img):
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Save label to custom-labels folder
            label_path = os.path.join("custom-labels", f"{filename}.txt")
            with open(label_path, "w") as f:
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {
                        width:.6f} {height:.6f}\n")

            # Save cropped image to custom-labels folder
            image_path = os.path.join("custom-labels", f"{filename}.jpg")
            cropped_img.save(image_path)

            st.success(
                f"Label and cropped image saved:\n- {label_path}\n- {image_path}")

        # Save cropped image only
        def save_cropped_image(image, filename):
            path = os.path.join("custom-labels", f"{filename}.jpg")
            image.save(path)
            st.success(f"Cropped image saved as {path}")
        st.divider()
        # Streamlit UI setup
        st.title("Annonation Tool")

        # Sidebar controls
        realtime_update = st.checkbox(
            "Update in Real Time", value=True)
        box_color = st.color_picker("Box Color", value='#0b4cd9')
        aspect_choice = st.radio("Aspect Ratio", options=[
            "1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1), "16:9": (
            16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None}
        aspect_ratio = aspect_dict[aspect_choice]

        # Upload image
        img_file = st.file_uploader(
            "Upload an Image", type=['png', 'jpg', 'jpeg'])
        
        # Placeholder for the progress bar
        progress_bar = st.progress(0)

        if img_file is not None:

            # Upload progress
            for percent_complete in range(0, 101, 10):
                time.sleep(0.1)  # Simulate upload delay
                progress_bar.progress(percent_complete)
            st.image(img_file, width=500, caption="Uploaded Image Successfully")
            progress_bar.empty()

        if img_file:
            try:
                # Load and display the uploaded image
                img = Image.open(img_file)
                img_width, img_height = img.size

                st.write("Original Image:")
                st.image(img, width=500)

                # Cropping using Streamlit cropper tool
                cropped_img = st_cropper(
                    img, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
                st.write("Cropped Image Preview:")
                st.image(cropped_img, width=500)

                # Input for label
                label = st.text_input(
                    "Enter Label for the bounding box and then hit enter:", "")

                # Label entered by the user
                # Display the raw value entered
                st.write(f"Entered Label: '{label}'")

                # Check if label is empty
                if not label:
                    st.error("Label cannot be empty.")
                else:
                    # Bounding box coordinates
                    left, upper, right, lower = cropped_img.getbbox()
                    bbox = (left, upper, right, lower)

                    # File saving buttons
                    filename = os.path.splitext(img_file.name)[0]
                    if st.button("Save Cropped Image"):
                        save_cropped_image(cropped_img, filename)

                    if st.button("Save YOLO Label"):
                        save_img_label_yolo_format(
                            filename, img_width, img_height, bbox, label, cropped_img)

            except Exception as e:
                st.error(f"An error occurred: {e}")
