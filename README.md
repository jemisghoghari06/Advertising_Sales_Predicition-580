# Advertising Sales Prediction 📈

Mini Project for **SYBCA-AI (Sem-III) | 303 Machine Learning**  
**Team Members:** Jemis Ghoghari, Parth Patel

---

## 📌 Project Overview
This project predicts advertising sales using machine learning techniques based on multi-channel marketing expenditures (TV, Radio, Newspaper, Digital media). It walks through all practical stages from exploratory data analysis to feature engineering, model training, evaluation, and fine-tuning.

---

## 📂 Project Structure

```
Advertising_Sales_Prediction/
├── Dataset/
│   ├── dataset.csv            # Original Indian advertising dataset
│   ├── dataset_clean.csv      # Cleaned data (post-practical 5)
│   └── split_data.pkl         # Train/test split data (post-practical 7)
├── Python_Code/
│   ├── practical_1_project_proposal.py
│   ├── practical_2_dataset_collection.py
│   ├── practical_3_understanding_data.py
│   ├── practical_4_eda.py
│   ├── practical_5_handle_missing_outliers.py
│   ├── practical_6_automated_eda.py
│   ├── practical_7_feature_selection.py
│   ├── practical_8_model_building.py
│   ├── practical_9_model_evaluation.py
│   ├── practical_10_improve_model.py
│   └── practical_11_results_conclusion.py
├── Reports/
│   └── EDA_Report.html        # Automated EDA report
├── .env.example               # Sample environment configuration template
├── .gitignore                 # Ignored files (secrets, venv, cache)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/jemisghoghari06/Advertising_Sales_Predicition-580.git
cd Advertising_Sales_Predicition-580
```

### 2. Set up virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Practical Execution Workflow

Run each script inside the `Python_Code/` directory in sequence:

```bash
cd Python_Code

python practical_1_project_proposal.py
python practical_2_dataset_collection.py
python practical_3_understanding_data.py
python practical_4_eda.py
python practical_5_handle_missing_outliers.py    # Generates dataset_clean.csv
python practical_6_automated_eda.py              # Generates EDA_Report.html
python practical_7_feature_selection.py          # Generates split_data.pkl
python practical_8_model_building.py             # Generates model_results.pkl
python practical_9_model_evaluation.py
python practical_10_improve_model.py
python practical_11_results_conclusion.py
```

---

## 🛡️ Best Practices & Security
- API keys and private tokens are never committed.
- Environment variables are isolated inside `.env` (ignored by `.gitignore`).
