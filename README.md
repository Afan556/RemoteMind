# 🧠 RemoteMind – Cognitive Load Tracker for Remote Workers

RemoteMind is a data science-driven burnout prediction dashboard designed to track the **cognitive load** of remote workers using behavioral signals like screen time, meetings, and breaks. Built with a focus on **mental health and workplace productivity**, it empowers organizations and individuals to proactively manage burnout risks.

---

## 🚀 Features

- 📈 Predicts stress levels using ML models trained on screen time & meeting patterns
- 📊 Interactive Streamlit dashboard with KPIs and visualizations
- 💬 Smart wellness suggestions based on user risk scores
- 🔄 Upload CSV to get real-time burnout insights

---

## 📁 Folder Structure

RemoteMind/
│
├── app/ # Streamlit dashboard
│ ├── app.py
│ └── components/
│ ├── burnout_dashboard.py
│ └── suggestions_generator.py
│
├── data/ # Raw datasets
│ └── Remote_Work_Cognitive_Stress.csv
│
├── models/ # Trained model
│ └── burnout_model.pkl
│
├── notebooks/ # EDA and experiments
│ └── 01_eda_visualization.ipynb
│
├── src/ # Feature & model logic
│ ├── data_preprocessing.py
│ ├── feature_extraction.py
│ └── model.py
│
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧪 How It Works

1. Upload a CSV file with remote work behavioral data
2. Features are engineered (call overload, break efficiency, etc.)
3. A trained regression model predicts stress levels
4. Results are visualized with wellness suggestions shown

---

## 🖥 Sample Data Format

| Avg_Screen_Time_Hrs | Breaks_Taken | Video_Call_Minutes | Reported_Stress_Level |
|---------------------|--------------|---------------------|------------------------|
| 6.5                 | 3            | 120                 | 7                      |

Download the sample dataset:  
[Remote_Work_Cognitive_Stress.csv](./data/Remote_Work_Cognitive_Stress.csv)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/RemoteMind.git
cd RemoteMind
pip install -r requirements.txt
streamlit run app/app.py