import streamlit as st
import pandas as pd
import joblib


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Weather Summary AI",
    page_icon="🌦",
    layout="wide"
)


# ---------------- FIX BACKGROUND ----------------

light_bg = "https://images.unsplash.com/photo-1534088568595-a066f410bcda"


st.markdown(
f"""
<style>

.stApp {{

background-image:

linear-gradient(
rgba(255,255,255,0.45),
rgba(255,255,255,0.45)
),

url("{light_bg}");

background-size:cover;

background-position:center;

background-attachment:fixed;

}}


.result-box {{

background:rgba(255,255,255,0.75);

padding:25px;

border-radius:20px;

text-align:center;

font-size:30px;

font-weight:bold;

}}

</style>

""",

unsafe_allow_html=True
)



# ---------------- LOAD MODEL ----------------


model = joblib.load(
    "weather_summary_classifier.pkl"
)


encoder = joblib.load(
    "label_encoder.pkl"
)



# ---------------- TITLE ----------------


st.title("🌦 Weather Summary Classification")

st.write(
"Enter weather details and predict summary"
)



# ---------------- INPUT ----------------


c1,c2,c3 = st.columns(3)



with c1:

    temp = st.number_input(
        "Temperature (C)",
        value=15.0
    )


    humidity = st.number_input(
        "Humidity",
        value=0.80
    )


    wind = st.number_input(
        "Wind Speed (km/h)",
        value=10.0
    )



with c2:


    apparent = st.number_input(
        "Apparent Temperature (C)",
        value=14.0
    )


    visibility = st.number_input(
        "Visibility (km)",
        value=10.0
    )


    pressure = st.number_input(
        "Pressure (millibars)",
        value=1010.0
    )



with c3:


    bearing = st.number_input(
        "Wind Bearing (degrees)",
        value=200
    )


    precip = st.selectbox(
        "Precip Type",
        [
            "rain",
            "snow"
        ]
    )



# ---------------- PREDICT ----------------


if st.button("🚀 Predict"):


    data = pd.DataFrame({


        "Formatted Date":
        [
            "2016-06-15 12:00:00"
        ],


        "Precip Type":
        [
            precip
        ],


        "Temperature (C)":
        [
            temp
        ],


        "Apparent Temperature (C)":
        [
            apparent
        ],


        "Humidity":
        [
            humidity
        ],


        "Wind Speed (km/h)":
        [
            wind
        ],


        "Wind Bearing (degrees)":
        [
            bearing
        ],


        "Visibility (km)":
        [
            visibility
        ],


        "Pressure (millibars)":
        [
            pressure
        ]

    })



    # Date conversion

    data["Formatted Date"] = pd.to_datetime(
        data["Formatted Date"]
    )


    data["Year"] = data["Formatted Date"].dt.year

    data["Month"] = data["Formatted Date"].dt.month

    data["Day"] = data["Formatted Date"].dt.day

    data["Hour"] = data["Formatted Date"].dt.hour

    data["DayOfWeek"] = data["Formatted Date"].dt.dayofweek



    data["Temp_Humidity"] = (

        data["Temperature (C)"]

        *

        data["Humidity"]

    )



    data["Wind_Visibility"] = (

        data["Wind Speed (km/h)"]

        /

        data["Visibility (km)"]

    )



    # Prediction

    prediction = model.predict(data)



    summary = encoder.inverse_transform(
        prediction
    )[0]



    # RESULT


    st.markdown(

    f"""

    <div class="result-box">

    🌦 Weather Summary

    <br><br>

    {summary}

    </div>

    """,

    unsafe_allow_html=True

    )