import streamlit as st
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
import cv2
import numpy as np

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    learn = load_learner(Path("C:\\Users\\SUNNY SANGWAN\\ML\\C-B-R\\data\\cattle\\export.pkl"))
    return learn

learn = load_model()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Cattle & Buffalo Breed Recognition", page_icon="🐄")
st.title("🐄 Cattle & Buffalo Breed Recognition")

uploaded_file = st.file_uploader("Upload an image of cattle or buffalo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(400, 400), caption="Uploaded Image", use_container_width=True)

    # Convert to numpy for face detection
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    # Breed prediction
    pred_class, pred_idx, probs = learn.predict(img)
    max_prob = probs[pred_idx].item()

    # -------------------------------
    # Decision Logic
    # -------------------------------
    if len(faces) > 0:
        st.error("🚫 Human detected! Please upload an image of cattle or buffalo.")
    elif max_prob < 0.50:   # lower threshold so cattle/buffalo aren’t rejected
        st.error("⚠️ This image does not appear to be a cattle or buffalo. Please upload a valid image.")
    else:
        st.success(f"✅ Predicted Breed: **{pred_class}** (Confidence: {max_prob:.2f})")
        st.write("Prediction probabilities:")
        for i, c in enumerate(learn.dls.vocab):
            st.write(f"- {c}: {probs[i]:.4f}")

# -------------------------------
# About Section (Sidebar)
# -------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    ### 🐄 Cattle & Buffalo Breed Recognition App  
    This application uses **FastAI + PyTorch** to recognize different breeds of cattle and buffalo.  

    - **Backend**: FastAI deep learning model trained on cattle & buffalo images.  
    - **Frontend**: Built with [Streamlit](https://streamlit.io).  
    - **Face Detection**: OpenCV Haar Cascade to block human uploads.  
    - **Confidence Filtering**: Ensures only valid cattle/buffalo predictions are shown.  

    ### 📌 Features
    - Upload cattle/buffalo image → Get breed prediction  
    - Human images → Blocked automatically  
    - Displays prediction probabilities for transparency  

    ⚡ *Developed for smart agriculture and breed recognition projects.*
    """
)
