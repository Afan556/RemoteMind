import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np # Often useful for RMSE

# --- 1. Load Data ---
# Load the dataset from the specified CSV file path.
# Ensure 'data/remote_mind_data_processed2.csv' exists in your project structure.
try:
    data = pd.read_csv(r'data/remote_mind_data_processed2.csv')
    print("Data loaded successfully.")
    print(f"Dataset shape: {data.shape}")
    print("Columns available:", data.columns.tolist())
except FileNotFoundError:
    print("Error: 'remote_mind_data_processed2.csv' not found. Please check the file path.")
    exit() # Exit if data cannot be loaded

# --- 2. Define Features (X) and Target (Y) ---
# Features are the independent variables used to predict the target.
# Target is the dependent variable we want to predict (burnout_index).
# Ensure these column names exactly match those in your CSV file.
features = ['Video_Call_Score', 'Break_Efficiency', 'Average_Screen_Time_Hours', 'Cognitive_Load_Index']
target = 'burnout_index'

# Check if all required columns exist in the DataFrame
if not all(col in data.columns for col in features + [target]):
    missing_cols = [col for col in features + [target] if col not in data.columns]
    print(f"Error: Missing required columns in the dataset: {missing_cols}")
    exit()

X = data[features]
Y = data[target]

print(f"\nFeatures selected: {features}")
print(f"Target selected: {target}")

# --- 3. Split Data into Training and Testing Sets ---
# Splitting the data ensures that we evaluate the model on unseen data,
# giving a more realistic estimate of its performance in the real world.
# test_size=0.2 means 20% of the data will be used for testing, 80% for training.
# random_state ensures reproducibility of the split.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"\nTraining set shape (X_train, Y_train): {X_train.shape}, {Y_train.shape}")
print(f"Test set shape (X_test, Y_test): {X_test.shape}, {Y_test.shape}")

# --- 4. Train and Evaluate Linear Regression Model ---
print("\n--- Training Linear Regression Model ---")
# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model using the training data
linear_model.fit(X_train, Y_train)

# Evaluate the model on the test data
# The .score() method for regression models returns the R-squared (R^2) score.
# R^2 represents the proportion of variance in the dependent variable that is predictable
# from the independent variables. Higher is better (closer to 1.0).
r2_lr = linear_model.score(X_test, Y_test)
print(f'Linear Regression Model R^2 score: {r2_lr:.4f}')

# Optional: Calculate additional metrics for Linear Regression
Y_pred_lr = linear_model.predict(X_test)
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
mse_lr = mean_squared_error(Y_test, Y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
print(f'Linear Regression MAE: {mae_lr:.4f}')
print(f'Linear Regression MSE: {mse_lr:.4f}')
print(f'Linear Regression RMSE: {rmse_lr:.4f}')

# --- 5. Train and Evaluate Random Forest Regressor Model ---
print("\n--- Training Random Forest Regressor Model ---")
# Initialize the Random Forest Regressor model
# n_estimators: The number of trees in the forest. More trees generally improve performance
#               but increase computation time. 100 is a good starting point.
# random_state: Ensures reproducibility of the results.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, Y_train)

# Evaluate the Random Forest model on the test data
r2_rf = rf_model.score(X_test, Y_test)
print(f'Random Forest Model R^2 score: {r2_rf:.4f}')

# Optional: Calculate additional metrics for Random Forest
Y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
mse_rf = mean_squared_error(Y_test, Y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print(f'Random Forest MAE: {mae_rf:.4f}')
print(f'Random Forest MSE: {mse_rf:.4f}')
print(f'Random Forest RMSE: {rmse_rf:.4f}')

# --- 6. Save the Chosen Model ---
# It's good practice to save the model that performs best or the one you intend to deploy.
# Here, we'll save the Random Forest model as it often performs better than Linear Regression
# on complex, real-world data due to its ability to capture non-linear relationships.
# If you prefer to save the Linear Regression model, change 'rf_model' to 'linear_model'.

model_to_save = rf_model # You can choose 'linear_model' if its performance is preferred or sufficient

# Define the path to save the model.
# Ensure the 'models' directory exists in your project.
model_filename = 'models/burnout_model.pkl' # Changed filename to reflect RF model


try:
    joblib.dump(model_to_save, model_filename)
    print(f"\nModel successfully saved to '{model_filename}'")
except Exception as e:
    print(f"\nError saving model: {e}")

# --- Optional: Load and Test the Saved Model ---
# This block demonstrates how to load a saved model and make predictions.
print("\n--- Demonstrating Model Loading and Prediction ---")
try:
    loaded_model = joblib.load(model_filename)
    print(f"Model successfully loaded from '{model_filename}'")

    # Make predictions on a sample from the test set
    sample_index = 0
    sample_features = X_test.iloc[[sample_index]]
    actual_burnout = Y_test.iloc[sample_index]
    predicted_burnout = loaded_model.predict(sample_features)

    print(f"\nSample Input Features:\n{sample_features.to_string(index=False)}")
    print(f"Actual Burnout Index: {actual_burnout:.4f}")
    print(f"Predicted Burnout Index: {predicted_burnout[0]:.4f}")

except FileNotFoundError:
    print(f"Error: Saved model '{model_filename}' not found for loading test.")
except Exception as e:
    print(f"Error during model loading/testing: {e}")