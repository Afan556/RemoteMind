from feature_extraction import create_features
import pandas as pd

data= pd.read_csv(r"data\remote_mind_data.csv")
data=create_features(data)
print(data.head())
