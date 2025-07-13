import streamlit as st
import pandas as pd # Explicitly import pandas for type hinting

# --- Function to Generate AI-Powered Wellness Suggestions ---
# This function takes a DataFrame (df) that should include
# the 'Predicted_Burnout_Index' column (as renamed from 'Predicted_Stress_Level'
# for consistency with the model's target).
def generate_suggestions(df: pd.DataFrame):
    # Set a subheader for the suggestions section in the Streamlit app.
    st.subheader("💡 AI Wellness Suggestions")

    # --- Check for 'Predicted_Burnout_Index' Column ---
    # It's good practice to ensure the necessary column exists before proceeding.
    if "Predicted_Burnout_Index" not in df.columns:
        st.error("Cannot generate suggestions: 'Predicted_Burnout_Index' column is missing from the data.")
        return # Exit the function if the column is not found

    # --- Identify High-Risk Individuals/Data Points ---
    # Filter the DataFrame to find rows where the predicted burnout index is 8 or higher.
    # This threshold (8) is an arbitrary example; you might adjust it based on
    # domain knowledge or the typical range of your burnout_index.
    high_risk_threshold = 8
    high_risk_entries = df[df["Predicted_Burnout_Index"] >= high_risk_threshold]

    # --- Provide Suggestions Based on Risk Level ---
    if not high_risk_entries.empty:
        # If there are any entries (rows) where burnout risk is high, display a warning
        st.warning(f"⚠️ High Burnout Risk Detected for {len(high_risk_entries)} individuals/data points!")
        st.write(f"Based on the analysis (Predicted Burnout Index >= {high_risk_threshold}):")
        
        # Display specific recommendations for high-risk scenarios.
        # These are general suggestions; you could make them more specific
        # by analyzing the features of the high_risk_entries (e.g., if high screen time
        # is common in high_risk, suggest specific screen time reductions).
        st.write("- **Recommend mandatory short breaks** every hour to prevent fatigue buildup.")
        st.write("- **Reduce meeting load** or encourage asynchronous communication to lower video call fatigue.")
        st.write("- **Encourage off-screen time** immediately post-work to promote mental disengagement.")
        st.write("- Suggest **mindfulness exercises** or short physical activities during breaks.")
        st.write("- Advise **reviewing work-life balance** and setting clear boundaries.")

    else:
        # If no entries meet the high-risk criteria, display a success message.
        st.success(f"✅ Burnout risk is currently within healthy levels (Predicted Burnout Index < {high_risk_threshold}).")
        st.write("Keep up the great work! Here are some tips to maintain well-being:")
        
        # Display recommendations for maintaining healthy levels.
        st.write("- **Maintain current break routine** and ensure they are truly restorative.")
        st.write("- **Continue managing optimal screen time** and focus on digital well-being.")
        st.write("- **Regularly assess** your mental state and workload to pre-empt potential stress.")
        st.write("- **Explore new wellness practices** to further enhance your well-being.")

    # --- Optional: More Granular Suggestions (Example) ---
    # You could add more nuanced suggestions based on other feature values.
    # For example, for individuals with high Cognitive_Load_Index even if burnout is not yet high:
    st.write("---") # Separator


    avg_cognitive_load = df["Cognitive_Load_Index"].mean()
    avg_screen_time = df["Average_Screen_Time_Hours"].mean()
    
    if avg_cognitive_load > df["Cognitive_Load_Index"].median() * 1.2: # Example threshold
        st.write(f"- Your average Cognitive Load Index ({avg_cognitive_load:.2f}) appears elevated. Consider techniques like task batching or focused work sessions.")

    if avg_screen_time > df["Average_Screen_Time_Hours"].median() * 1.5: # Example threshold
        st.write(f"- Your average daily screen time ({avg_screen_time:.2f} hours) is on the higher side. Try incorporating screen-free activities or using blue light filters.")

    # You could also show suggestions based on specific feature values for individual data points
    # (e.g., if you had an ID for each person/period):
    # for index, row in high_risk_entries.iterrows():
    #     st.write(f"--- Recommendations for Data Point/Individual ID {index} (Burnout: {row['Predicted_Burnout_Index']:.2f}) ---")
    #     if row['Video_Call_Score'] > X_train['Video_Call_Score'].mean(): # Compare to training data mean
    #         st.write("- High video call score detected. Try reducing non-essential video meetings.")
    #     # ... and so on for other features