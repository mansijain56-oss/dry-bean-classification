# Dry Bean Classification – Machine Learning Assignment 2

## BITS Pilani Work Integrated Learning Programme

**Course:** AIMLCZG565 – Machine Learning  
**Assignment:** 2  
**Student:** Mansi Jain  
**Student ID:** 2025AC05151  
**Programme:** M.Tech AI & ML  

---

## 🔗 Quick Links

- 🌐 **Live Streamlit Application:** https://dry-bean-classification-6paakea9eqlqmwkdm5dgje.streamlit.app/
- 💻 **GitHub Repository:** https://github.com/mansijain56-oss/dry-bean-classification
- 📊 **UCI Dry Bean Dataset:** https://archive.ics.uci.edu/dataset/602/dry+bean
- 📚 **Dataset Research Paper:** https://doi.org/10.1016/j.compag.2020.105507
- 📘 **Scikit-learn Documentation:** https://scikit-learn.org/stable/
- 🚀 **Streamlit Documentation:** https://docs.streamlit.io/

---

## 1. Project Overview

This project implements an end-to-end multiclass machine learning solution for classifying dry bean varieties using morphological and geometric features.

The project covers:

- Data understanding and exploratory data analysis
- Data quality validation
- Data preprocessing
- Stratified train/test split
- Feature scaling
- Multiple supervised classification models
- Hyperparameter tuning
- Model evaluation and comparison
- Confusion matrix and classification report
- Feature importance analysis
- Single-record prediction
- Batch prediction
- Interactive Streamlit application
- Streamlit Community Cloud deployment

---

## 2. Problem Statement

The objective is to classify dry bean samples into one of seven bean classes using 16 numerical morphological features.

The problem is formulated as a multiclass supervised classification task.

---

## 3. Dataset Description

| Property | Value |
|---|---|
| Dataset | Dry Bean Dataset |
| Source | UCI Machine Learning Repository |
| Instances | 13,611 |
| Features | 16 numerical features |
| Classes | 7 |
| Missing Values | None |
| Problem Type | Multiclass Classification |

### Bean Classes

Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira

### Features

Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

---

## 4. Data Preprocessing

1. Dataset quality checks
2. Missing-value and duplicate checks
3. Target label encoding
4. Stratified train/test split
5. StandardScaler feature scaling
6. Fitting preprocessing objects using the training data
7. Saving the fitted scaler and label encoder for reuse during prediction

---

## 5. Exploratory Data Analysis

The project includes dataset structure and summary statistics, class distribution, numerical feature distributions, boxplots and violin plots, correlation analysis, feature relationships, outlier analysis and class-wise feature behaviour.

The detailed EDA is available in the project notebooks.

---

## 6. Machine Learning Models

Five supervised classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

Hyperparameter tuning was performed using GridSearchCV where applicable.

---

## 7. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

For multiclass ROC-AUC evaluation, the One-vs-Rest approach is used.

Confusion matrices and classification reports are also generated.

---

## 8. Model Comparison

Final results are available in:

`output/model_comparison.csv`

The Streamlit application provides:

- Model performance table
- Best-performing model indicators
- Bar chart
- Radar chart
- Model-specific confusion matrix
- Classification report

The confusion matrix updates according to the selected model.

---

## 9. Feature Importance

Feature importance is calculated using the trained Random Forest model.

Results are available in:

`output/feature_importance.csv`

The application provides feature importance ranking, scores, visualization, cumulative importance, an 80% cumulative-importance reference and the top three important features.

Feature importance represents the model's contribution to impurity-based split decisions and should not be interpreted as causal influence.

---

## 10. Streamlit Application

The application contains:

### 🏠 Home
Project overview, dataset summary, model summary, objectives and workflow.

### 📊 Dataset Explorer
Dataset preview, statistics, class distribution, correlation analysis, feature distributions and data-quality information.

### 🤖 Single Prediction
Enter the 16 feature values, select a trained model and generate the predicted bean class, confidence, top-3 predicted classes and class probabilities.

### 📁 Batch Prediction
Upload CSV data, validate the required feature columns, generate predictions, view the prediction summary and predicted class distribution.

### 📈 Model Comparison
Interactive metrics table, best-model indicators, bar chart, radar chart, model-specific confusion matrix and classification report.

### 🌳 Feature Importance
Random Forest feature importance ranking and visualizations.

### ℹ️ About
Project, student, methodology and technology information.

---

## 11. Model Persistence

The trained models and preprocessing objects are saved using Joblib.

```text
model/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── scaler.pkl
└── label_encoder.pkl
```

---

## 12. Repository Structure

```text
dry-bean-classification/
│
├── app.py
├── README.md
├── requirements.txt
├── Dry_Bean_Dataset.xlsx
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── output/
│   ├── train_processed.csv
│   ├── test_data.csv
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── ML_Assignment2_Models.ipynb
└── train_drybean_models.ipynb
```

---

## 13. Requirements and Local Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 14. Deployment

### Live Streamlit Application

https://dry-bean-classification-6paakea9eqlqmwkdm5dgje.streamlit.app/

### GitHub Repository

https://github.com/mansijain56-oss/dry-bean-classification

The repository contains the Streamlit application, notebooks, trained models, preprocessing objects, output files and requirements file.

---

## 15. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- OpenPyXL

---

## 16. Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
EDA
   ↓
Data Quality Checks
   ↓
Preprocessing
   ↓
Stratified Train/Test Split
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
Confusion Matrix & Classification Report
   ↓
Feature Importance
   ↓
Single & Batch Prediction
   ↓
Streamlit Deployment
```

---

## 17. Results Summary

The final application compares five classification models using Accuracy, Precision, Recall, F1 Score, MCC and ROC-AUC.

The saved model evaluation results are available in:

`output/model_comparison.csv`

The application provides an interactive comparison of the models and allows selection of a model to inspect its confusion matrix and classification report.

---

## 18. Conclusion

This project demonstrates an end-to-end machine learning workflow for multiclass dry bean classification.

Five supervised learning algorithms were implemented and evaluated using multiple classification metrics. The models were compared using quantitative evaluation measures, confusion matrices and classification reports. Random Forest feature importance was used to identify influential features.

The trained models and preprocessing objects were persisted and integrated into an interactive Streamlit application supporting both single-record and batch predictions.

The final application is deployed through Streamlit Community Cloud and the complete project is maintained in the GitHub repository.

---

## 19. References

1. **UCI Machine Learning Repository – Dry Bean Dataset**  
   https://archive.ics.uci.edu/dataset/602/dry+bean

2. **Koklu, M. and Ozkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.**  
   https://doi.org/10.1016/j.compag.2020.105507

3. **Scikit-learn Documentation**  
   https://scikit-learn.org/stable/

4. **Streamlit Documentation**  
   https://docs.streamlit.io/

5. **BITS Pilani WILP – AIMLCZG565 Machine Learning Assignment 2**

---

## Student Details

| Field | Details |
|---|---|
| Name | Mansi Jain |
| Student ID | 2025AC05151 |
| Course | AIMLCZG565 – Machine Learning |
| Programme | M.Tech AI & ML |
| University | BITS Pilani WILP |

---

## Final Project Links

**Live App:** https://dry-bean-classification-6paakea9eqlqmwkdm5dgje.streamlit.app/

**GitHub:** https://github.com/mansijain56-oss/dry-bean-classification

**UCI Dataset:** https://archive.ics.uci.edu/dataset/602/dry+bean
