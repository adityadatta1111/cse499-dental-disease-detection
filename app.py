from pathlib import Path
import PIL
import PIL.Image
import streamlit as st
import settings
import helper

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

confidence = float(st.sidebar.slider(
    "Confidence Threshold (In Percentage %)", 25, 50, 100
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image Successfully")
        except Exception as ex:
            st.error(f"Error loading image: {ex}")
            
    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path
            )
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Disease and Stuff'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                try:
                    with st.expander("Detectioin Results"):
                        for box in boxes:
                            st.write(box.data)
                            class_label = model.names[int(box.cls)]
                            st.write(class_label)
                except Exception as ex:
                    st.write("No image has been uploaded.")
