import streamlit as st
import requests
from PIL import Image

# Backend API URL
API_URL = "https://facescope-api.onrender.com/verify"

st.title("Face Recognition System")
st.write("Upload a CROPPED face image for verification.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare file for API
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")
    }

    try:
        # Send request to backend
        response = requests.post(API_URL, files=files, timeout=30)

        if response.status_code == 200:
            data = response.json()

            st.write(f"### Best Match: {data['best_match']}")
            st.write(f"### Similarity Score: {data['similarity_score']:.4f}")

            if data["result"] == "VERIFIED":
                st.success("VERIFIED")
            else:
                st.error("UNKNOWN PERSON")

        else:
            st.error("API Error: Unable to get response")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API")

    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
