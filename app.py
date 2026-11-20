import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image


# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# Configure Streamlit page
st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide"
)


# Application title
st.title("📄 Smart Document Scanner")

st.write(
    "Upload a document image and extract text using OCR."
)


# Image upload
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)


if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    img_array = np.array(image)

    # Display original image
    st.subheader("Original Document")

    st.image(
        image,
        use_container_width=True
    )

    # Image preprocessing
    gray = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Display processed image
    st.subheader("Processed Document")

    st.image(
        thresh,
        clamp=True,
        use_container_width=True
    )

    # OCR button
    if st.button("Extract Text"):

        with st.spinner("Extracting text..."):

            extracted_text = pytesseract.image_to_string(
                thresh,
                lang="eng"
            )

        # Display extracted text
        st.subheader("Extracted Text")

        st.text_area(
            "OCR Result",
            extracted_text,
            height=300
        )

        if extracted_text.strip():

            # Download extracted text
            st.download_button(
                label="Download Extracted Text",
                data=extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        else:
            st.warning("No text detected in the image.")
