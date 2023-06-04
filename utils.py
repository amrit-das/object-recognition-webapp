# Import Needed Packages
import cv2
import numpy as np
import streamlit as st

# Import Yolo Packages
from ultralytics import YOLO


def load(model_path):
    return YOLO(model_path)


# def display_tracker_options():
#     display_tracker = st.sidebar.radio("Display Tracker", ("Yes", "No"))
#     is_display_tracker = True if display_tracker == "Yes" else False
#     if is_display_tracker:
#         tracker_type = st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
#         return is_display_tracker, tracker_type
#     return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (640, 480))
    # if is_display_tracking:
    #     res = model.track(image, conf=conf, persist=True, tracker=tracker)
    # else:
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted, caption="Detections", channels="BGR", use_column_width=False
    )


def from_webcam(conf, model):
    # is_display_tracker, tracker = display_tracker_options()
    try:
        if st.sidebar.button("Start Recognition"):
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
    except Exception as e:
        st.sidebar.error("Error Occured")
        st.sidebar.error(e)


def from_image(conf, model):
    img_src_up = st.sidebar.file_uploader(
        "Upload Image", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    if st.sidebar.button("Uplaod"):
        file_bytes = np.asarray(bytearray(img_src_up.read()), dtype=np.uint8)
        img_src = cv2.imdecode(file_bytes, 1)
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                img_src,
                caption="Uploaded Image",
                use_column_width=True,
                channels="BGR",
            )

        with col2:
            res = model.predict(img_src, conf=conf)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(
                res_plotted,
                caption="Detected Image",
                use_column_width=True,
            )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")
