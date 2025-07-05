#  Vehicle Insurance Claim Fraud Detection

This project uses machine learning to detect fraudulent vehicle insurance claims. The dataset contains various claim details such as policyholder demographics, vehicle information, and accident specifics. The project employs preprocessing, class balancing, and XGBoost classification to achieve high fraud detection accuracy.

---

##  Project Objective

Develop a predictive model to classify whether a vehicle insurance claim is fraudulent, helping insurance companies reduce losses and improve claims processing.

---

##  Dataset

* Source: Kaggle (Oracle Vehicle Insurance Claim Dataset)
* File: `fraud_oracle.csv`
* Records: \~\[Insert number of rows]
* Features: Include `Make`, `AccidentArea`, `Sex`, `VehicleAge`, `AnnualPremium`, etc.
* Target: `FraudFound_P` or equivalent binary fraud indicator

---

##  Tech Stack

* **Language:** Python (Google Colab)
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **ML Techniques:**

  * Label Encoding
  * Class balancing with SMOTE
  * XGBoost Classification

---

## Workflow

1. Load and inspect dataset
2. Handle missing values and encode categorical variables
3. Define target and features
4. Balance classes using SMOTE
5. Train XGBoost model
6. Evaluate model (Accuracy, Precision, Recall, F1-Score)
7. Visualize feature importances

---

##  Results

* Accuracy: 97%
* High Recall & Precision for fraud class
* Balanced detection using SMOTE-enhanced data

---

##  Files Included

* `fraud_oracle.csv` – Input dataset
* `fraud_detection.ipynb` – Colab notebook with full pipeline
* `fraud_model_smote.pkl` – Trained model
* `README.md` – Project description

---

##  Future Enhancements

* Hyperparameter tuning with GridSearchCV
* Web application using Streamlit
* SHAP/LIME explainability modules

---

##  How to Run

1. Upload the dataset and notebook to Google Colab
2. Run the notebook cells sequentially
3. Review evaluation outputs and feature importance
4. Save model for deployment or further use

---

> Developed by Abhijit Kolekar.
