import streamlit as st
import pandas as pd
import joblib
import os # To check for file existence more robustly

# --- Import custom modules ---
# Assuming 'src' is the parent directory for 'feature_extraction.py'
# and 'components' is a sibling directory.
# Ensure your project structure is:
# your_project/
# ├── dashboard.py
# ├── src/
# │   └── feature_extraction.py
# │   └── __init__.py  (Important for src to be a package)
# ├── components/
# │   └── burnout_dashboard.py
# │   └── suggestions_generator.py
# │   └── __init__.py  (Important for components to be a package)
# └── models/
#     └── burnout_model.pkl
# └── data/
#     └── remote_mind_data_processed2.csv (if used for training script)

try:
    # Attempt to import feature_extraction from src.
    # This assumes 'src' is directly accessible in the Python path
    # (e.g., you're running streamlit from the project root).
    from feature_extraction import create_features
except ImportError as e:
    st.error(f"Error importing 'src.feature_extraction': {e}. "
            "Ensure 'src' is a Python package (has an __init__.py file) "
            "and your Streamlit app is run from the project's root directory.")
    st.stop() # Stop execution if a critical import fails

try:
    # Attempt to import dashboard display component
    from components.burnout_dashboard import display_dashboard
    # Attempt to import suggestions generator component
    from components.suggestions_generator import generate_suggestions
except ImportError as e:
    st.error(f"Error importing 'components': {e}. "
            "Ensure 'components' is a Python package (has an __init__.py file) "
            "and your Streamlit app is run from the project's root directory.")
    st.stop() # Stop execution if a critical import fails


# --- Streamlit Page Configuration ---
# Sets the browser tab title, icon, and overall layout of the Streamlit app.
st.set_page_config(
    page_title="Remote Mind Burnout Tracker",
    page_icon=":brain:",
    layout="centered" # Can be "wide" for more content width
)

# --- Application Title ---
st.title("🧠 RemoteMind - Cognitive Load Tracker for Remote Workers")

# --- File Uploader Widget ---
# Allows users to upload their CSV data file.
uploaded_file = st.file_uploader("Upload your data file (CSV)", type=["csv"])

# --- Main Logic for File Processing and Prediction ---
if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame.
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully. Processing data...")
        
        # --- Feature Engineering ---
        # Call the custom function to create necessary features from the raw data.
        # This function should ensure the DataFrame 'df' has the columns
        # expected by the model (Video_Call_Score, Break_Efficiency, etc.).
        processed_df = create_features(df.copy()) # Use .copy() to avoid SettingWithCopyWarning later

        # --- Load Pre-trained Model ---
        # Define the path to the saved model.
        model_path = "models/burnout_model.pkl"

        # Check if the model file exists before attempting to load it.
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at '{model_path}'. "
                    "Please ensure the 'models' folder and 'burnout_model.pkl' exist in your project root.")
            st.stop() # Stop execution if the model isn't found
            
        model = joblib.load(model_path)
        st.info("Machine learning model loaded.")

        # --- Make Predictions ---
        # Ensure the features used for prediction match the order and names
        # the model was trained on. This is crucial!
        required_features = ["Video_Call_Score", "Break_Efficiency", "Average_Screen_Time_Hours", "Cognitive_Load_Index"]

        # Check if all required features exist in the processed DataFrame
        if not all(feature in processed_df.columns for feature in required_features):
            missing = [f for f in required_features if f not in processed_df.columns]
            st.error(f"Error: Processed data is missing required features for prediction: {missing}. "
                    "Please check the `create_features` function.")
            st.stop()
            
        # Predict burnout index using the loaded model.
        # Ensure that the input data 'processed_df[required_features]' is a DataFrame,
        # not a Series, as the model expects a 2D array-like input.
        processed_df["Predicted_Burnout_Index"] = model.predict(processed_df[required_features])
        
        st.success("Burnout index predictions completed successfully!")
        
        # --- Display Dashboard ---
        # Call the component to visualize the data and predictions.
        display_dashboard(processed_df)
        
        # --- Generate Suggestions ---
        # Call the component to provide personalized suggestions based on predictions.
        generate_suggestions(processed_df)
        
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
    except pd.errors.ParserError:
        st.error("Could not parse the CSV file. Please ensure it is a valid CSV format.")
    except KeyError as ke:
        st.error(f"A required column was not found after feature creation: {ke}. "
                "Check `create_features` output and your model's expected input features.")
    except Exception as e:
        # Catch any other unexpected errors during processing or prediction.
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e) # Display full traceback for debugging

else:
    # Message displayed when no file is uploaded yet.
    st.info("Please upload a CSV file to begin predicting burnout levels.")