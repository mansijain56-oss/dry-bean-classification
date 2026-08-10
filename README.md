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

## 2. Assignment Deliverables

This repository contains the major components required for the assignment:

- Data preprocessing and exploratory data analysis
- Train/test data split
- Multiple machine learning classification models
- Hyperparameter tuning
- Model evaluation using Accuracy, Precision, Recall, F1 Score, MCC and ROC-AUC
- Confusion matrices and classification reports
- Model comparison
- Feature importance analysis
- Saved trained models and preprocessing objects
- Single-record prediction
- Batch prediction
- Interactive Streamlit application
- Jupyter notebooks containing the implementation
- Requirements file for environment setup
- Project documentation

---

## 3. Problem Statement

The objective is to classify dry bean samples into one of seven bean classes using 16 numerical morphological features.

The problem is formulated as a multiclass supervised classification task.

---

## 4. Dataset Description

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

- Seker
- Barbunya
- Bombay
- Cali
- Dermason
- Horoz
- Sira

### Features

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

---

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Dataset quality checks
2. Missing-value and duplicate checks
3. Target label encoding
4. Stratified train/test split
5. StandardScaler feature scaling
6. Fitting preprocessing objects using the training data
7. Saving the fitted scaler and label encoder for reuse during prediction

The same saved preprocessing objects are used by the Streamlit application for prediction.

---

## 6. Exploratory Data Analysis

The project includes:

- Dataset structure and summary statistics
- Class distribution
- Numerical feature distributions
- Boxplots and violin plots
- Correlation analysis
- Feature relationships
- Outlier analysis
- Class-wise feature behaviour

The detailed EDA is available in the project notebooks.

---

## 7. Machine Learning Models

Five supervised classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

Hyperparameter tuning was performed using GridSearchCV where applicable.

---

## 8. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

For multiclass ROC-AUC evaluation, the One-vs-Rest approach is used.

Confusion matrices and classification reports are also generated for model evaluation.

---

## 9. Model Comparison

Final model comparison results are available in:

```text
output/model_comparison.csv
```

The Streamlit application provides:

- Model performance table
- Best-performing model indicators
- Bar chart
- Radar chart
- Model-specific confusion matrix
- Classification report

The confusion matrix updates according to the selected model.

---

## 10. Feature Importance

Feature importance is calculated using the trained Random Forest model.

Results are available in:

```text
output/feature_importance.csv
```

The Streamlit application provides:

- Feature importance ranking
- Importance scores
- Feature importance visualization
- Cumulative feature importance
- 80% cumulative-importance reference
- Top three important features

Feature importance represents the model's contribution to impurity-based split decisions and should not be interpreted as causal influence.

---

## 11. Streamlit Application

The application contains the following sections:

### 🏠 Home

Provides the project overview, dataset summary, model summary, objectives and workflow.

### 📊 Dataset Explorer

Provides dataset preview, statistics, class distribution, correlation analysis, feature distributions and data-quality information.

### 🤖 Single Prediction

Users can enter the 16 feature values, select a trained model and generate:

- Predicted bean class
- Prediction confidence
- Top-3 predicted classes
- Prediction probabilities for all classes
- Prediction probability visualization

### 📁 Batch Prediction

Users can upload CSV data and generate predictions for multiple records.

The page provides:

- Uploaded data preview
- Required-column validation
- Batch predictions
- Prediction summary
- Predicted class distribution
- Prediction results

### 📈 Model Comparison

Provides:

- Evaluation metrics table
- Best Accuracy
- Best F1 Score
- Best ROC-AUC
- Bar chart
- Radar chart
- Model-specific confusion matrix
- Classification report

The confusion matrix is generated for the selected model using the held-out test data.

### 🌳 Feature Importance

Provides Random Forest feature importance ranking and visualizations.

### ℹ️ About

Provides project, student, methodology, technology and application information.

---

## 12. Model Persistence

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

These saved objects are loaded by the Streamlit application for prediction and model analysis.

---

## 13. Repository Structure

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

## 14. Requirements and Local Execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at the local Streamlit URL displayed in the terminal.

---

## 15. Deployment

### 🌐 Live Streamlit Application

https://dry-bean-classification-6paakea9eqlqmwkdm5dgje.streamlit.app/

### 💻 GitHub Repository

https://github.com/mansijain56-oss/dry-bean-classification

The repository contains the Streamlit application, notebooks, trained models, preprocessing objects, output files and requirements file.

---

## 16. Technologies Used

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

## 17. Project Workflow

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

## 18. Results Summary

The final application compares five classification models using Accuracy, Precision, Recall, F1 Score, MCC and ROC-AUC.

The saved model evaluation results are available in:

```text
output/model_comparison.csv
```

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9935 | 0.9224 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9071 | 0.9696 | 0.9073 | 0.9071 | 0.9071 | 0.8876 |
| K-Nearest Neighbors | 0.9174 | 0.9849 | 0.9182 | 0.9174 | 0.9175 | 0.9001 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Random Forest | 0.9214 | 0.9924 | 0.9215 | 0.9214 | 0.9214 | 0.9049 |

### Model Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Achieved the joint-highest accuracy (92.14%) and the highest AUC (0.9935). Its strong generalization, simple structure, and fast inference make it a strong choice for this dataset. |
| Decision Tree | Achieved 90.71% accuracy and 0.9696 AUC. Its performance is lower than the other models, although it provides interpretable decision rules. |
| K-Nearest Neighbors | Achieved 91.74% accuracy and 0.9849 AUC, showing strong classification and class-separation performance. |
| Gaussian Naive Bayes | Achieved 89.79% accuracy and 0.9902 AUC. The high AUC indicates strong class-separation capability despite the lower classification accuracy. |
| Random Forest | Achieved the joint-highest accuracy (92.14%) with strong performance across all metrics. It also provides useful feature-importance information. |
| **Overall Winner** | **Logistic Regression** is the overall winner for this dataset because it matches Random Forest on accuracy while achieving the highest AUC (0.9935), along with strong Precision, Recall, F1 and MCC. |

---

## 19. Conclusion

This project demonstrates an end-to-end machine learning workflow for multiclass dry bean classification.

Five supervised learning algorithms were implemented and evaluated using multiple classification metrics. The models were compared using quantitative evaluation measures, confusion matrices and classification reports. Random Forest feature importance was used to identify influential features.

The trained models and preprocessing objects were persisted and integrated into an interactive Streamlit application supporting both single-record and batch predictions.

The final application is deployed through Streamlit Community Cloud and the complete project is maintained in the GitHub repository.

---

## 20. References

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
| **Name** | Mansi Jain |
| **Student ID** | 2025AC05151 |
| **Course** | AIMLCZG565 – Machine Learning |
| **Programme** | M.Tech AI & ML |
| **University** | BITS Pilani WILP |

---

## Final Project Links

🌐 **Live App:**  
https://dry-bean-classification-6paakea9eqlqmwkdm5dgje.streamlit.app/

💻 **GitHub:**  
https://github.com/mansijain56-oss/dry-bean-classification

📊 **UCI Dataset:**  
https://archive.ics.uci.edu/dataset/602/dry+bean
