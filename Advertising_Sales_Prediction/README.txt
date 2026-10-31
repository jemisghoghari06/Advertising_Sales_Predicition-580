============================================================
  SYBCA-AI (Sem-III) | 303 Machine Learning
  Mini Project: Advertising Sales Prediction
  Members:-jemis ghoghari,parth patel
============================================================

HOW TO RUN (in order):
-----------------------
Step 1:  cd Python_Code
Step 2:  pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling sweetviz

Run each practical file in order:
  python practical_1_project_proposal.py
  python practical_2_dataset_collection.py
  python practical_3_understanding_data.py
  python practical_4_eda.py
  python practical_5_handle_missing_outliers.py   ← generates dataset_clean.csv
  python practical_6_automated_eda.py             ← generates EDA_Report.html
  python practical_7_feature_selection.py         ← generates split_data.pkl
  python practical_8_model_building.py            ← generates model_results.pkl
  python practical_9_model_evaluation.py
  python practical_10_improve_model.py
  python practical_11_results_conclusion.py

FOLDER STRUCTURE:
-----------------
Advertising_Sales_Prediction/
├── Dataset/
│   ├── dataset.csv            ← original 550-record Indian dataset
│   ├── dataset_clean.csv      ← generated after Practical 5
│   └── split_data.pkl         ← generated after Practical 7
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
│   └── EDA_Report.html        ← generated after Practical 6
└── README.txt
============================================================
