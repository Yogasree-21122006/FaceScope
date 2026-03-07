import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/verify"

st.title("Face Recognition System")
st.write("Upload a CROPPED face image for verification.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()

        st.write(f"### Best Match: {data['best_match']}")
        st.write(f"### Similarity Score: {data['similarity_score']:.4f}")

        if data["result"] == "VERIFIED":
            st.success("VERIFIED")
        else:
            st.error("UNKNOWN PERSON")

    else:
        st.error("API Error")

