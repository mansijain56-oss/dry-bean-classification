# Dry Bean Classification - ML Assignment 2

## BITS Pilani Work Integrated Learning Programme

**Course:** AIMLCZG565 – Machine Learning  
**Student:** Mansi Jain (2025AC05151)  
**Programme:** M.Tech AI & ML

---

## 1. Problem Statement

The objective of this project is to classify dry bean varieties based on their geometric and shape features. Using 16 morphological attributes extracted from images of seven different bean types, five supervised machine learning classification models are implemented and compared to determine an effective approach for accurate bean variety identification.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| **Dataset** | Dry Bean Dataset |
| **Source** | UCI Machine Learning Repository |
| **Instances** | 13,611 |
| **Features** | 16 numerical features |
| **Classes** | 7 |
| **Missing Values** | None |
| **Problem Type** | Multi-class Classification |

### Classes

Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira

### Features

Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

### Preprocessing

- Stratified train/test split
- Label encoding of the target variable
- StandardScaler applied to feature variables
- Fitted encoder and scaler saved as Joblib `.pkl` files

---

## 3. GitHub Repository

[GitHub Repository](https://github.com/mansijain56-oss/dry-bean-classification)

### Repository Structure

```text
dry-bean-classification/
├── app.py
├── requirements.txt
├── README.md
├── Dry_Bean_Dataset.xlsx
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── output/
│   ├── train_processed.csv
│   ├── test_data.csv
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── ML_Assignment2_Models.ipynb
└── train_drybean_models.ipynb
```

---

## 4. Models Used

Five supervised classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

Hyperparameter tuning was performed using GridSearchCV where applicable.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

Confusion matrices and model comparison results are also generated.

---

## 5. Model Comparison

The final model comparison is stored in:

```text
output/model_comparison.csv
```

The file contains the performance metrics for all five models.

The Streamlit application provides an interactive model comparison page with:

- Metrics table
- Model ranking
- Bar chart
- Radar chart
- Confusion matrix
- Classification report

---

## 6. Feature Importance

Random Forest feature importance is calculated from the trained Random Forest model.

The results are saved to:

```text
output/feature_importance.csv
```

The Streamlit application displays feature importance scores, rankings, a feature importance chart, cumulative importance, and the top three features.

---

## 7. Streamlit Application

The project includes an interactive Streamlit web application.

### Application Pages

- **Home** – Project overview, dataset statistics, class distribution and best-model summary
- **Dataset Explorer** – Preview, statistics, distributions, correlation heatmap, histograms and feature information
- **Single Prediction** – Predict a bean variety from 16 feature values and view confidence where supported
- **Batch Prediction** – Upload a CSV, generate predictions and download the results
- **Model Comparison** – Compare all five models across the evaluation metrics
- **Feature Importance** – Explore Random Forest feature importance
- **About** – Project and student information

---

## 8. Saved Model Files

The `model/` directory contains:

```text
logistic_regression.pkl
decision_tree.pkl
knn.pkl
naive_bayes.pkl
random_forest.pkl
scaler.pkl
label_encoder.pkl
```

These saved objects are loaded by the Streamlit application for prediction and evaluation.

---

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/mansijain56-oss/dry-bean-classification.git
cd dry-bean-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 10. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open using the local Streamlit URL.

---

## 11. Streamlit Deployment

The application is designed for deployment using Streamlit Community Cloud from the GitHub repository.

**Repository:** `mansijain56-oss/dry-bean-classification`  
**Main file:** `app.py`

**Live Streamlit App:** To be added after deployment.

---

## 12. Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- OpenPyXL

---

## 13. Project Workflow

```text
Dry Bean Dataset
       ↓
Data Exploration
       ↓
Data Preprocessing
       ↓
Label Encoding
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Feature Importance
       ↓
Save Models as .pkl
       ↓
Streamlit Deployment
```

---

## 14. References

1. Koklu, M. and Ozkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. *Computers and Electronics in Agriculture*, 174, 105507.
2. UCI Machine Learning Repository – Dry Bean Dataset.
3. Scikit-learn Documentation.
4. Streamlit Documentation.

---

## Student Details

| Field | Details |
|---|---|
| **Name** | Mansi Jain |
| **Student ID** | 2025AC05151 |
| **Course** | AIMLCZG565 – Machine Learning |
| **Programme** | M.Tech AI & ML |
| **University** | BITS Pilani WILP |
