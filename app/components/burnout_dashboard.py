import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def display_dashboard(df):
    st.subheader("📊 Dashboard Insights")

    st.metric("Average Predicted Stress", round(df["Predicted_Stress_Level"].mean(), 2))
    st.metric("Max Predicted Stress", round(df["Predicted_Stress_Level"].max(), 2))
    st.metric("Average Cognitive Load Index", round(df["Cognitive_Load_Index"].mean(), 2))

    st.write("### Stress Level Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Predicted_Stress_Level"], kde=True, bins=10, ax=ax, color="crimson")
    st.pyplot(fig)

    st.write("### Cognitive Load vs Predicted Stress")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="Cognitive_Load_Index", y="Predicted_Stress_Level", hue="breaks_taken", palette="coolwarm", ax=ax2)
    st.pyplot(fig2)
    st.write("### Video Call Score vs Predicted Stress")
    fig3, ax3 = plt.subplots()