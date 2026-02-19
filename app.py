import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wildlife Classifier",
    page_icon="🌲",
    layout="centered"
)

# ---------------- UI CSS ----------------
st.markdown("""
<style>

/* Remove Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Brighter forest background with lighter overlay */
.stApp {
    background: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main glass card */
.main-card {
    background: rgba(0,0,0,0.55);
    padding: 40px;
    border-radius: 25px;
    box-shadow: 0 0 30px rgba(0,255,150,0.4);
    text-align: center;
    margin-top: 60px;
}

/* Title */
.title-text {
    font-size: 48px;
    font-weight: 900;
    color: #00ff9d;
    text-shadow: 0 0 18px #00ff9d, 2px 2px 12px black;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle-text {
    font-size: 20px;
    color: white;
    margin-bottom: 30px;
    text-shadow: 2px 2px 8px black;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #00ff9d;
}

/* Remove accidental empty div blocks */
div:empty {
    display: none;
}

/* Prediction result */
.prediction-text {
    font-size: 28px;
    font-weight: bold;
    color: #00ff9d;
    margin-top: 20px;
    text-shadow: 0 0 15px #00ff9d, 2px 2px 10px black;
}

/* Confidence text */
.confidence-text {
    font-size: 18px;
    color: white;
    margin-top: 10px;
}

/* Progress bar color */
.stProgress > div > div > div > div {
    background-color: #00ff9d;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "animal_classifier_model.h5")
model = load_model(model_path)

# ---------------- CLASS NAMES ----------------
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe",
    "Horse", "Kangaroo", "Lion", "Panda",
    "Tiger", "Zebra"
]

# ---------------- EMOJI MAP ----------------
animal_emojis = {
    "Bear": "🐻",
    "Bird": "🐦",
    "Cat": "🐱",
    "Cow": "🐄",
    "Deer": "🦌",
    "Dog": "🐶",
    "Dolphin": "🐬",
    "Elephant": "🐘",
    "Giraffe": "🦒",
    "Horse": "🐴",
    "Kangaroo": "🦘",
    "Lion": "🦁",
    "Panda": "🐼",
    "Tiger": "🐯",
    "Zebra": "🦓"
}

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title-text">🌲 Wildlife Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload an image and explore the animal kingdom</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an Animal Image", type=["jpg", "png", "jpeg"])

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: match training preprocessing
    # img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    emoji = animal_emojis.get(predicted_class, "🐾")

    st.markdown(
        f'<div class="prediction-text">{emoji} {predicted_class}</div>',
        unsafe_allow_html=True
    )

    st.progress(confidence)

    st.markdown(
        f'<div class="confidence-text">Confidence: {confidence*100:.2f}%</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
