import streamlit as st

def generate_suggestions(df):
    st.subheader("💡 AI Wellness Suggestions")

    high_risk = df[df["Predicted_Stress_Level"] >= 8]
    if not high_risk.empty:
        st.warning("⚠️ High Burnout Risk Detected!")
        st.write("- Recommend mandatory short breaks every hour.")
        st.write("- Reduce meeting load or switch to async communication.")
        st.write("- Encourage off-screen time post-work.")
    else:
        st.success("✅ Burnout risk is within healthy levels.")
        st.write("- Keep current break routine.")
        st.write("- Maintain optimal screen time and mental balance.")
