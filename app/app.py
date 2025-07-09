import streamlit as st 
import pandas as pd
import joblib
from feature_extraction import create_features
from components.burnout_dashboard import display_dashboard
from components.suggestions_generator import generate_suggestions

#title
st.set_page_config(page_title="Remote Mind Burnout Tracker", page_icon=":brain:", layout="centered")
st.title("🧠 RemoteMind - Cognitive Load Tracker for Remote Workers")

#file uploader
uploaded_file= st.file_uploader("Upload your data file (CSV)", type=["csv"])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    try:
        df = create_features(df)
        model = joblib.load("models/burnout_model.pkl")
        df["Predicted_Stress_Level"] = model.predict(df[["Video_Call_Score", "Break_Efficiency", "Average_Screen_Time_Hours", "Cognitive_Load_Index"]])
        
        st.success("Predictions completed successfully!")
        display_dashboard(df)
        generate_suggestions(df)
        
    except Exception as e:
        st.error(f"Something went wrong: {e}")
