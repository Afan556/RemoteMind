import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sns
data = pd.read_csv(r"data\remote_mind_data.csv")
numeric_cols= data.select_dtypes(include='number').columns
for col in numeric_cols:
    pt.figure(figsize=(12,4))
    pt.subplot(1,2,1)
    sns.histplot(data[col], kde=True)
    pt.title(f"{col} Distribution")
    
    pt.subplot(1,2,2)
    sns.boxplot(x=data[col])
    pt.title(f"{col} Boxplot")
    pt.show()

#Bivariate Analysis (Correlations)
data_numeric = data.select_dtypes(include=['number'])
pt.figure(figsize=(8,6))
sns.heatmap(data_numeric.corr(),annot=True, cmap="coolwarm")
pt.title("Feature correlation Heatmap")
pt.show()
#pairplot
sns.pairplot(data,diag_kind='kde')
pt.suptitle("Future pair Relationships",y=1.02)
pt.show()

# Grouped Insights
data.groupby("breaks_taken")["reported_stress_level"].mean().plot(kind="bar")
pt.title("Average Stress Level by Number of Breaks")
pt.ylabel("Mean Stress Level")
pt.xlabel("Breaks Taken")
pt.show()
