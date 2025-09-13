import streamlit as st
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
import cv2
import numpy as np
from datetime import datetime

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    learn = load_learner(Path("C:\\Users\\SUNNY SANGWAN\\Cattle-Buffalo-Breed-Recognition-APP\\data\\cattle\\export.pkl"))
    return learn

learn = load_model()

# -------------------------------
# Breed Information Dictionary
# -------------------------------
breed_info = {
    "Murrah": "Origin: Haryana, India 🐃 | High milk yield (8–16 L/day) | Jet black body | Long tightly curved horns.",
    "Jaffarabadi": "Origin: Gujarat, India 🐃 | Large heavy breed | Used for both milk and draught | Long flat drooping horns.",
    "Nili-Ravi": "Origin: Punjab (India & Pakistan) 🐃 | Popular dairy buffalo | Milk yield 10–15 L/day | Wall eyes (white marks around eyes).",
    "Surti": "Origin: Gujarat 🐃 | Moderate milk yield | Curved sickle-shaped horns | Compact body size.",
    "Jersey": "Origin: Jersey Island 🐄 | Light brown color | High butterfat content in milk (4.8–5.2%) | Small and docile breed.",
    "Holstein Friesian": "Origin: Netherlands 🐄 | Black & white patches | Highest milk yield (20–30 L/day) | Large body size.",
    "Gir": "Origin: Gujarat 🐄 | Red with white patches | Known for A2 milk | Long pendulous ears.",
    "Sahiwal": "Origin: Punjab 🐄 | Reddish-brown color | High heat tolerance | Milk yield 8–10 L/day.",
    "Red Sindhi": "Origin: Sindh, Pakistan 🐄 | Deep reddish color | Good milk producer | Dual-purpose (milk + draught)."
}

# -------------------------------
# Load Haar Cascade for human face detection
# -------------------------------
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

    # Show current date & time
    current_time = datetime.now().strftime("%A, %d %B %Y %I:%M %p")
    st.info(f"📅 Uploaded on: **{current_time}**")

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
        st.error("🚫 This is not Cattle or Buffalo! Please upload an image of Cattle or Buffalo.")
    elif max_prob < 0.20:   # Lower threshold for recognition
        st.warning("⚠️ Low confidence prediction. This might still be a cattle/buffalo, but the breed is uncertain.")
        st.info(f"Closest Match: **{pred_class}** (Confidence: {max_prob:.2f})")
    else:
        st.success(f"✅ Predicted Breed: **{pred_class}** (Confidence: {max_prob:.2f})")

        # Show probabilities
        st.subheader("📊 Prediction Probabilities")
        for i, c in enumerate(learn.dls.vocab):
            st.write(f"- {c}: {probs[i]:.4f}")

        # Show breed qualities if available
        if pred_class in breed_info:
            st.subheader("📌 Breed Qualities")
            st.write(breed_info[pred_class])
        else:
            st.info("ℹ️ Breed details not available. You may add this breed to the dictionary.")

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
    - Shows current **date & time** of upload  
    - Displays **breed qualities** (milk yield, origin, physical traits, etc.)  

    ⚡ *Developed for smart agriculture and breed recognition projects.*  
    """
)
