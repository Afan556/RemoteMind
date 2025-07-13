import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Explicitly import pandas, though it's likely already in the main app

# --- Function to Display Dashboard Insights ---
# This function takes a DataFrame (df) which is expected to contain
# processed data, including the 'Predicted_Burnout_Index' (previously 'Predicted_Stress_Level')
# and other relevant features like 'Cognitive_Load_Index', 'Video_Call_Score', and 'Break_Efficiency'.
def display_dashboard(df: pd.DataFrame):
    # Set a subheader for the dashboard section in the Streamlit app.
    st.subheader("📊 Dashboard Insights")

    # --- Display Key Metrics ---
    # st.metric displays a key-value pair, useful for showing single, important numbers.
    # We use .mean() and .max() to get aggregate statistics for the predicted stress and cognitive load.
    # .round(2) formats the numbers to two decimal places for readability.

    # Display the average predicted burnout index.
    # Note: The main app code uses "Predicted_Stress_Level" but the model is for "burnout_index".
    # I'll stick to "Predicted_Burnout_Index" for consistency with the model's target.
    # If your model truly predicts "Stress_Level", adjust the column name here.
    if "Predicted_Burnout_Index" in df.columns:
        st.metric("Average Predicted Burnout Index", round(df["Predicted_Burnout_Index"].mean(), 2))
        # Display the maximum predicted burnout index.
        st.metric("Max Predicted Burnout Index", round(df["Predicted_Burnout_Index"].max(), 2))
    else:
        st.warning("Predicted_Burnout_Index column not found. Cannot display predicted stress metrics.")

    # Display the average Cognitive Load Index from the input data.
    if "Cognitive_Load_Index" in df.columns:
        st.metric("Average Cognitive Load Index", round(df["Cognitive_Load_Index"].mean(), 2))
    else:
        st.warning("Cognitive_Load_Index column not found. Cannot display average cognitive load.")


    # --- Stress Level Distribution Plot ---
    st.write("### Burnout Index Distribution")
    # Create a Matplotlib figure and an axes object for the plot.
    fig, ax = plt.subplots(figsize=(10, 6)) # Set a specific figure size for better display
    
    # Use Seaborn's histplot to visualize the distribution of predicted burnout index.
    # kde=True adds a Kernel Density Estimate curve for smoother distribution visualization.
    # bins=10 sets the number of bins for the histogram.
    # ax=ax specifies that the plot should be drawn on our created axes.
    # color="crimson" sets the color of the histogram bars.
    if "Predicted_Burnout_Index" in df.columns:
        sns.histplot(df["Predicted_Burnout_Index"], kde=True, bins=10, ax=ax, color="crimson")
        ax.set_title("Distribution of Predicted Burnout Index") # Add a title to the plot
        ax.set_xlabel("Predicted Burnout Index") # Label the x-axis
        ax.set_ylabel("Frequency") # Label the y-axis
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        st.pyplot(fig) # Display the Matplotlib figure in Streamlit.
        plt.close(fig) # Close the figure to free up memory (important for Streamlit apps).
    else:
        st.info("Cannot display Burnout Index Distribution: 'Predicted_Burnout_Index' column is missing.")


    # --- Cognitive Load vs Predicted Burnout Plot ---
    st.write("### Cognitive Load vs Predicted Burnout Index")
    # Create another Matplotlib figure and axes for this scatter plot.
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Use Seaborn's scatterplot to show the relationship between Cognitive Load Index
    # and Predicted Burnout Index.
    # hue="Break_Efficiency" colors the points based on 'Break_Efficiency',
    # allowing us to see if break efficiency influences this relationship.
    # palette="coolwarm" sets the color scheme for the hue.
    if "Cognitive_Load_Index" in df.columns and "Predicted_Burnout_Index" in df.columns and "Break_Efficiency" in df.columns:
        sns.scatterplot(data=df, x="Cognitive_Load_Index", y="Predicted_Burnout_Index",
                        hue="Break_Efficiency", palette="coolwarm", ax=ax2)
        ax2.set_title("Cognitive Load vs Predicted Burnout Index (by Break Efficiency)")
        ax2.set_xlabel("Cognitive Load Index")
        ax2.set_ylabel("Predicted Burnout Index")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Cannot display Cognitive Load vs Predicted Burnout: One or more required columns are missing.")


    # --- Video Call Score vs Predicted Burnout Plot ---
    st.write("### Video Call Score vs Predicted Burnout Index")
    # Create the third Matplotlib figure and axes.
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Similar to the previous scatter plot, visualize the relationship between
    # Video Call Score and Predicted Burnout Index.
    # We can use 'Break_Efficiency' as hue again, or choose another relevant feature
    # like 'Average_Screen_Time_Hours' if it makes sense for visual analysis.
    # For now, let's use 'Average_Screen_Time_Hours' to show another dimension.
    if "Video_Call_Score" in df.columns and "Predicted_Burnout_Index" in df.columns and "Average_Screen_Time_Hours" in df.columns:
        sns.scatterplot(data=df, x="Video_Call_Score", y="Predicted_Burnout_Index",
                        hue="Average_Screen_Time_Hours", palette="viridis", ax=ax3) # Using 'viridis' palette
        ax3.set_title("Video Call Score vs Predicted Burnout Index (by Avg Screen Time)")
        ax3.set_xlabel("Video Call Score")
        ax3.set_ylabel("Predicted Burnout Index")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Cannot display Video Call Score vs Predicted Burnout: One or more required columns are missing.")

    # You can add more plots here as needed to visualize other relationships
    # or feature distributions relevant to burnout.
    # Example:
    # st.write("### Break Efficiency Distribution")
    # fig4, ax4 = plt.subplots()
    # sns.histplot(df["Break_Efficiency"], kde=True, bins=10, ax=ax4, color="teal")
    # st.pyplot(fig4)
    # plt.close(fig4)