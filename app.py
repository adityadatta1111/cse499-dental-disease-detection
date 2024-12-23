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

# Initialize session states
if 'img_file' not in st.session_state:
    st.session_state.img_file = None

if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None

if 'label' not in st.session_state:
    st.session_state.label = ""

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
img_file = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image.", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

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

                st.session_state.img_file = uploaded_image  # Save the uploaded image to session state
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
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot(labels=False)[:, :, ::-1]
                st.session_state.img_file = Image.fromarray(res_plotted)  # Update session state with detected image
                st.session_state.img_file.save("detected_image.jpg")
                Image.open("detected_image.jpg")

    with st.container():
        
        # Check `custom-labels` folder exists
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
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

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
        st.title("Annotation Tool")

        # Sidebar controls
        realtime_update = st.checkbox(
            "Update in Real Time", value=True)
        box_color = st.color_picker("Box Color", value='#0b4cd9')
        aspect_choice = st.radio("Aspect Ratio", options=[
            "1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1), "16:9": (
            16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None}
        aspect_ratio = aspect_dict[aspect_choice]

        if st.session_state.img_file:
            try:
                img = "/Users/adityadatta/Documents/programming/cse499-dental-disease-detection/detected_image.jpg"
                img_file = Image.open(img)
                img_width, img_height = img_file.size

                st.header("Predicted Image:")
                st.image(img_file, width=300)

                st.header("Edit Bounding Boxes:")
                # Cropping using Streamlit cropper tool
                st.session_state.cropped_img = st_cropper(
                    img_file, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
                st.header("Cropped Image Preview:")
                st.image(st.session_state.cropped_img, width=300)

                # Input for label
                st.header("Enter Label for the bounding box and then hit enter:")
                st.session_state.label = st.text_input("Enter Label here: ", st.session_state.label)
                st.header(f"Entered Label:'{st.session_state.label}'")

                # Check if label is empty
                if not st.session_state.label:
                    st.error("Label cannot be empty.")
                else:
                    # Bounding box coordinates
                    left, upper, right, lower = st.session_state.cropped_img.getbbox()
                    bbox = (left, upper, right, lower)

                    # File saving buttons
                    # Use `source_img.name` to get the file name for saving
                    filename = os.path.splitext(source_img.name)[0]
                    if st.button("Save Cropped Image"):
                        save_cropped_image(st.session_state.cropped_img, filename)

                    if st.button("Save YOLO Label"):
                        save_img_label_yolo_format(
                            filename, img_width, img_height, bbox, st.session_state.label, st.session_state.cropped_img)

            except Exception as e:
                st.error(f"An error occurred: {e}")
