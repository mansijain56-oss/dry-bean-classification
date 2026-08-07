# Dry Bean Classification - ML Assignment 2

## BITS Pilani Work Integrated Learning Programme
**Course:** AIMLCZG565 – Machine Learning  
**Student:** Mansi Jain (2025AC05151)  
**Programme:** M.Tech AI & ML  

---

## a. Problem Statement

The objective of this project is to classify dry bean varieties based on their geometric and shape features. Using 16 morphological attributes extracted from images of seven different bean types, we implement and compare five supervised machine learning classification models to determine the most effective approach for accurate bean variety identification.

---

## b. Dataset Description

| Property | Value |
|----------|-------|
| **Dataset** | Dry Bean Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| **Instances** | 13,611 |
| **Features** | 16 (all numeric) |
| **Classes** | 7 (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) |
| **Missing Values** | None |
| **Problem Type** | Multi-class Classification |

### Features
Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

### Train/Test Split
- Training: 90%
- Testing: 10%
- Preprocessing: StandardScaler applied to all features

---

## c. GitHub Repository Link

🔗 [GitHub Repository](https://github.com/YOUR_USERNAME/dry-bean-classification)

### Repository Structure
```
dry-bean-classification/
│-- app.py                    # Streamlit web application
│-- requirements.txt          # Python dependencies
│-- README.md                 # This file
│-- test_data.csv             # Test dataset used in experiments
│-- model_comparison.csv      # Model evaluation results
│-- Dry_Bean_Dataset.xlsx     # Original dataset
│-- model/                    # Saved model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│-- output/                   # Output files
│   └── model_comparison.csv
│-- ML_Assignment2_Models.ipynb      # EDA & analysis notebook
│-- train_drybean_models.ipynb       # Model training notebook
```

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9214 | 0.9935 | 0.9224 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9071 | 0.9696 | 0.9073 | 0.9071 | 0.9071 | 0.8876 |
| KNN | 0.9174 | 0.9849 | 0.9182 | 0.9174 | 0.9175 | 0.9001 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Random Forest (Ensemble) | 0.9214 | 0.9924 | 0.9215 | 0.9214 | 0.9214 | 0.9049 |

---

### Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the joint-highest accuracy (92.14%) with strong AUC (0.9935). Performs well as a baseline linear model despite the multi-class nature of the problem. Fast training and inference, good generalization. |
| **Decision Tree** | Lower accuracy (90.71%) compared to other models due to overfitting on training data. However, it provides interpretable decision rules. The AUC (0.9696) is the lowest among all models. |
| **KNN** | Good performance (91.74%) with balanced precision and recall. Slightly lower AUC (0.9849) than Logistic Regression. Performance is sensitive to the choice of K and distance metric. |
| **Naive Bayes** | Lowest accuracy (89.79%) due to the independence assumption not holding well for correlated geometric features. However, AUC (0.9902) is surprisingly high, indicating good probability calibration. |
| **Random Forest (Ensemble)** | Joint-highest accuracy (92.14%) with Logistic Regression. Provides feature importance insights and is robust against overfitting through bagging. Slightly lower AUC than Logistic Regression but most balanced across all metrics. |
| **Overall Winner** | **Logistic Regression** is the overall winner for this dataset — it matches Random Forest on accuracy while achieving the highest AUC (0.9935), indicating superior probability estimation. Its simplicity, fast inference, and strong generalization make it the best choice for production deployment on this dataset. |

---

## e. Live Streamlit App

🔗 [Live App on Streamlit Cloud](https://YOUR_APP_URL.streamlit.app)

---

## f. How to Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dry-bean-classification.git
cd dry-bean-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## g. Streamlit App Features

1. **Dataset Explorer** — View data statistics, distributions, correlations, and histograms
2. **Single Prediction** — Enter feature values to predict bean variety with confidence scores
3. **Batch Prediction** — Upload CSV file for bulk predictions with downloadable results
4. **Model Comparison** — Compare all 5 models across 6 metrics with bar charts and radar plots
5. **Confusion Matrix** — View classification performance per class for each model
6. **Feature Importance** — Visualize Random Forest feature importance with cumulative plot

---

## h. Technologies Used

- Python 3.11
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib (model serialization)

---

## i. References

1. Koklu, M. and Ozkan, I.A., (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 174, 105507.
2. UCI Machine Learning Repository - Dry Bean Dataset
3. Scikit-learn Documentation - https://scikit-learn.org/
4. Streamlit Documentation - https://docs.streamlit.io/
