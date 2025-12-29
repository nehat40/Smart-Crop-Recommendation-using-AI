import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Load dataset
df = pd.read_csv("Crop_recommendation5000.csv")

# Load model & scaler
model = pickle.load(open("crop01", "rb"))
scaler = pickle.load(open("scaler01", "rb"))

def navigate_to(page):
    st.session_state.current_page = page

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

with st.sidebar:
    st.title("Navigation")
    st.button("Home", on_click=navigate_to, args=("Home",))
    st.button("Dataset Overview", on_click=navigate_to, args=("Dataset Overview",))
    st.button("Crop Recommendation", on_click=navigate_to, args=("Crop Recommendation",))

if st.session_state.current_page == "Home":
    st.header("🌾 CROP RECOMMENDATION SYSTEM")


    image = Image.open("Presentation1.jpg")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style='text-align:center;'>
            <img src='data:image/jpeg;base64,{img_str}' width='600'/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "This system recommends the **best crop to grow** based on soil nutrients "
        "(Nitrogen, Phosphorus, Potassium) and environmental conditions like "
        "temperature, humidity, pH, and rainfall."
    )

elif st.session_state.current_page == "Dataset Overview":
    st.title("📊 Dataset Overview")

    st.markdown("**N** – Nitrogen content in soil")
    st.markdown("**P** – Phosphorus content in soil")
    st.markdown("**K** – Potassium content in soil")
    st.markdown("**Temperature** – Average temperature (°C)")
    st.markdown("**Humidity** – Relative humidity (%)")
    st.markdown("**pH** – Soil pH value")
    st.markdown("**Rainfall** – Rainfall (mm)")
    st.markdown("**Label** – Recommended Crop")

    if st.toggle("Show Dataset"):
        st.dataframe(df)

elif st.session_state.current_page == "Crop Recommendation":
    st.title("🌱 Crop Recommendation Checker")

    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)

    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    if st.button("🌾 PREDICT CROP"):
        features = [[N, P, K, temperature, humidity, ph, rainfall]]
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        st.success(f"✅ Recommended Crop: **{prediction[0].upper()}**")
        st.balloons()

