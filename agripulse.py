import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
import joblib
import pandas as pd
import xgboost
@st.cache_resource
def load_model():
    return joblib.load("agripulse.pkl")  
model = load_model()
def fetch_weather(lat=-6.2, lon=106.8):  
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m"
        ],
        "hourly": [
            "precipitation",
            "shortwave_radiation"
        ],
        "timezone": "auto"
    }

    responses = client.weather_api(url, params=params)
    response = responses[0]

    # --- Current values ---
    current = response.Current()
    temperature = current.Variables(0).Value()
    humidity = current.Variables(1).Value()

    # --- Hourly values (latest hour) ---
    hourly = response.Hourly()
    precipitation = hourly.Variables(0).ValuesAsNumpy()[-1]
    solar = hourly.Variables(1).ValuesAsNumpy()[-1]

    # --- Month ---
    month = datetime.now().month

    return {
        "rainfall_mm": float(precipitation),
        "temperature_c": float(temperature),
        "humidity_pct": float(humidity),
        "solar_irradiance_wm2": float(solar) if solar is not None else 0.0,
        "month": month
    }

import streamlit as st

st.title("AgripulseAI")
lcol1, lcol2 = st.columns(2)
with lcol1:
    lat = st.number_input("Latitude", value=-6.2)
    lon = st.number_input("Longitude", value=106.8)

with lcol2:
    yield_ton_per_ha = st.number_input("Threshold perkiraan hasil (ton/ha)", value=4)

if st.button("Fetch Weather Data"):
    weather = fetch_weather(lat, lon)
    st.session_state.weather = weather

weather = st.session_state.get("weather", {
    "rainfall_mm": 3,
    "temperature_c": 25.0,
    "humidity_pct": 60.0,
    "solar_irradiance_wm2": 0.0,
    "month": 1
})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cuaca")
    st.write("Data ini berasal dari BMKG (otomatis), dapat diubah untuk menyesuaikan")

    rainfall_mm = st.number_input(
        "Rainfall (mm)",
        value=weather["rainfall_mm"]
    )

    temperature_c = st.number_input(
        "Temperature (°C)",
        value=weather["temperature_c"]
    )

    humidity_pct = st.slider(
        "Humidity (%)",
        0, 100,
        int(weather["humidity_pct"])
    )

    solar_irradiance_wm2 = st.number_input(
        "Solar Irradiance (W/m²)",
        value=weather["solar_irradiance_wm2"]
    )

    month = weather["month"]


with col2:
    st.subheader("Data Sensor")
    st.write("Data ini berasal dari sensor Agripulse, untuk simulasi dapat dimasukan secara manual")


    light_intensity_lux = st.number_input(
        "Light Intensity (lux)",
        value=5000.0
    )

    soil_moisture_pct = st.slider(
        "Soil Moisture (%)",
        0, 100, 50
    )

    shading_pct = st.slider(
        "Shading (%)",
        0, 100, 20
    )

    co2_ppm = st.number_input(
        "CO2 (ppm)",
        value=400.0
    )

# --- Feature vector ---
feature = [
    rainfall_mm,
    light_intensity_lux,
    soil_moisture_pct,
    humidity_pct,
    temperature_c,
    month,
    solar_irradiance_wm2,
    shading_pct,
    co2_ppm,
    yield_ton_per_ha
]



if st.button("Predict"):
    try:
        input_df = pd.DataFrame([feature], columns=[
            "rainfall_mm", "light_intensity_lux",
            "soil_moisture_pct", "humidity_pct",
            "temperature_c", "month",
            "solar_irradiance_wm2",
            "shading_pct", "co2_ppm","yield_ton_per_ha"
        ])

        prediction = model.predict(input_df)
        st.subheader("Prediction Result")
        st.success("Prediction completed successfully!")
        tb = pd.DataFrame({
            "irrigation_liters": [prediction[0][0]],
            "panel_angle_deg": [prediction[0][1]],
            "fertilizer_kg": [prediction[0][2]],
        })
        st.dataframe(tb)
   

    except Exception as e:
        st.error(f"Prediction failed: {e}")