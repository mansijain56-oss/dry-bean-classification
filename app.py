import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }

    .metric-card h2 {
        font-size: 2rem;
        margin: 0;
    }

    .metric-card p {
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.85;
    }

    .prediction-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 25px;
        border-radius: 14px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 15px 0;
    }

    .welcome-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 22px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
    }

    .section-header {
        border-left: 5px solid #667eea;
        padding-left: 12px;
        margin: 20px 0 10px 0;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: #f5f7fb;
        margin: 10px 0;
    }

    .subtle-note {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: -6px;
    }

    .result-label {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 2px;
    }

    div[data-testid="stMetric"] {
        padding: 6px 4px;
    }

    .stDownloadButton button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION / USER INFORMATION
# ==========================================================

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

current_datetime = datetime.now(ZoneInfo("Asia/Kolkata"))
current_date = current_datetime.strftime("%d %B %Y")
current_time = current_datetime.strftime("%I:%M %p")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    "🎓 **BITS Pilani WILP**",
    unsafe_allow_html=True
)

st.sidebar.title("🌱 Dry Bean Classifier")
st.sidebar.caption(
    "AIMLCZG565 – Machine Learning | Assignment 2"
)
st.sidebar.caption("Application version: Final")

st.sidebar.divider()

user_name = st.sidebar.text_input(
    "👤 Your Name (Optional)",
    value=st.session_state.user_name,
    placeholder="Enter your name (optional)"
)

st.session_state.user_name = user_name.strip()

if st.session_state.user_name:
    st.sidebar.success(
        f"Welcome, {st.session_state.user_name}!"
    )
else:
    st.sidebar.success("Welcome!")

st.sidebar.caption(f"📅 {current_date}")
st.sidebar.caption(f"🕐 {current_time}")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "🤖 Single Prediction",
        "📁 Batch Prediction",
        "📈 Model Comparison",
        "🌳 Feature Importance",
        "ℹ️ About"
    ]
)

st.sidebar.divider()


# ==========================================================
# FEATURE NAMES
# ==========================================================

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

TARGET = "Class"


# ==========================================================
# LOAD DATASET & MODELS
# ==========================================================

@st.cache_data
def load_dataset():
    return pd.read_excel(
        "Dry_Bean_Dataset.xlsx",
        engine="openpyxl"
    )


@st.cache_resource
def load_models():
    model_dir = "model"

    models = {
        "Logistic Regression": joblib.load(
            os.path.join(model_dir, "logistic_regression.pkl")
        ),
        "Decision Tree": joblib.load(
            os.path.join(model_dir, "decision_tree.pkl")
        ),
        "KNN": joblib.load(
            os.path.join(model_dir, "knn.pkl")
        ),
        "Naive Bayes": joblib.load(
            os.path.join(model_dir, "naive_bayes.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(model_dir, "random_forest.pkl")
        )
    }

    scaler = joblib.load(
        os.path.join(model_dir, "scaler.pkl")
    )

    encoder = joblib.load(
        os.path.join(model_dir, "label_encoder.pkl")
    )

    return models, scaler, encoder


# ==========================================================
# LOAD APPLICATION RESOURCES
# ==========================================================

try:
    df = load_dataset()
    models, scaler, encoder = load_models()
except Exception as e:
    st.error(f"Unable to load project resources: {e}")
    st.stop()


# ==========================================================
# BASIC VALIDATION
# ==========================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features or TARGET not in df.columns:
    st.error(
        "Dataset structure is not as expected. "
        f"Missing columns: {missing_features}"
    )
    st.stop()


# ==========================================================
# SIDEBAR STATS
# ==========================================================

st.sidebar.markdown("**Dataset Stats**")
st.sidebar.markdown(f"- Records: **{df.shape[0]:,}**")
st.sidebar.markdown(f"- Features: **{len(FEATURES)}**")
st.sidebar.markdown(
    f"- Classes: **{df[TARGET].nunique()}**"
)
st.sidebar.markdown(
    f"- Models: **{len(models)}**"
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def get_model_comparison():
    """Load and validate model comparison results."""
    comparison = pd.read_csv(
        "output/model_comparison.csv"
    )

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC",
        "ROC-AUC"
    ]

    missing = [
        column
        for column in required_columns
        if column not in comparison.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in model_comparison.csv: "
            + ", ".join(missing)
        )

    for metric in required_columns[1:]:
        comparison[metric] = pd.to_numeric(
            comparison[metric],
            errors="coerce"
        )

    return comparison


def evaluate_model(model, X_test, y_test):
    """Calculate standard classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y_test,
            y_pred
        )
    }

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)

        classes = np.arange(
            len(encoder.classes_)
        )

        y_bin = label_binarize(
            y_test,
            classes=classes
        )

        try:
            metrics["ROC-AUC"] = roc_auc_score(
                y_bin,
                y_prob,
                multi_class="ovr",
                average="weighted"
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🌱 Dry Bean Classification System")

    if st.session_state.user_name:
        welcome_heading = f"👋 Welcome, {st.session_state.user_name}!"
    else:
        welcome_heading = "👋 Welcome!"

    st.markdown(
        f"""
        <div class="welcome-box">
            <h2>{welcome_heading}</h2>
            <p>Interactive Dry Bean Classification application</p>
            <p>📅 {current_date} &nbsp; | &nbsp; 🕐 {current_time}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "##### BITS Pilani WILP · AIMLCZG565 Machine Learning · Assignment 2"
    )

    st.divider()

    # Dynamic dashboard metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f'<div class="metric-card"><h2>{len(df):,}</h2>'
        '<p>Records</p></div>',
        unsafe_allow_html=True
    )

    c2.markdown(
        f'<div class="metric-card"><h2>{len(FEATURES)}</h2>'
        '<p>Features</p></div>',
        unsafe_allow_html=True
    )

    c3.markdown(
        f'<div class="metric-card"><h2>{df[TARGET].nunique()}</h2>'
        '<p>Bean Classes</p></div>',
        unsafe_allow_html=True
    )

    c4.markdown(
        f'<div class="metric-card"><h2>{len(models)}</h2>'
        '<p>ML Models</p></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Best model summary
    try:
        comparison = get_model_comparison()

        best_accuracy = comparison["Accuracy"].max()

        best_models = comparison.loc[
            comparison["Accuracy"] == best_accuracy,
            "Model"
        ].tolist()

        best_auc = comparison["ROC-AUC"].max()

        best_auc_models = comparison.loc[
            comparison["ROC-AUC"] == best_auc,
            "Model"
        ].tolist()

        c1, c2 = st.columns(2)

        with c1:
            st.success(
                f"🏆 Highest Accuracy: "
                f"**{', '.join(best_models)}** "
                f"({best_accuracy:.4f})"
            )

        with c2:
            st.info(
                f"📊 Highest ROC-AUC: "
                f"**{', '.join(best_auc_models)}** "
                f"({best_auc:.4f})"
            )

    except Exception:
        st.info(
            "Model comparison results are available "
            "on the Model Comparison page."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-header">🎯 Project Objectives</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        - Explore and analyze the Dry Bean Dataset
        - Train and evaluate five ML classifiers
        - Compare model performance across six metrics
        - Analyze feature importance
        - Deploy an interactive prediction interface
        """)

        st.markdown(
            '<div class="section-header">🤖 Models Used</div>',
            unsafe_allow_html=True
        )

        for model_name in models:
            st.markdown(f"- {model_name}")

    with col2:

        st.markdown(
            '<div class="section-header">📊 Class Distribution</div>',
            unsafe_allow_html=True
        )

        counts = df[TARGET].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        colors = sns.color_palette(
            "viridis",
            len(counts)
        )

        ax.barh(
            counts.index,
            counts.values,
            color=colors
        )

        ax.set_xlabel("Count")
        ax.invert_yaxis()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.markdown(
        '<div class="section-header">🔄 Project Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = """
    Dataset → EDA → Preprocessing → Train/Test Split →
    Feature Scaling → Model Training → Hyperparameter Tuning →
    Evaluation → Model Comparison → Feature Importance →
    Streamlit Deployment
    """

    st.info(workflow)

    st.caption(
        "The detailed implementation code is maintained in the "
        "project notebooks and app.py repository files."
    )


# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "📊 Dataset Explorer":

    st.title("📊 Dataset Explorer")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", len(FEATURES))
    c4.metric(
        "Classes",
        df[TARGET].nunique()
    )

    st.divider()

    # Dataset quality summary
    st.markdown(
        '<div class="section-header">🔎 Dataset Quality</div>',
        unsafe_allow_html=True
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    numeric_features = df[
        FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    infinite_values = int(
        np.isinf(
            numeric_features.to_numpy()
        ).sum()
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

    q2.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )

    q3.metric(
        "Infinite Values",
        f"{infinite_values:,}"
    )

    if missing_values == 0 and duplicate_rows == 0:
        st.success(
            "✅ Dataset quality check passed: no missing or infinite values detected."
        )
    elif missing_values == 0:
        st.warning(
            f"⚠️ No missing values detected, but "
            f"{duplicate_rows:,} duplicate rows were found."
        )
    else:
        st.warning(
            "⚠️ The dataset contains missing or duplicate values."
        )

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Preview",
        "📈 Distribution",
        "🔥 Correlation",
        "📉 Histograms",
        "📝 Feature Info"
    ])

    with tab1:

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.divider()

        st.markdown("**Summary Statistics**")

        st.dataframe(
            df.describe().T.style.background_gradient(
                cmap="Blues"
            ),
            use_container_width=True
        )

    with tab2:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.countplot(
            data=df,
            x=TARGET,
            order=df[TARGET].value_counts().index,
            palette="viridis",
            ax=ax
        )

        ax.set_title(
            "Bean Class Distribution"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        for patch in ax.patches:

            ax.annotate(
                f"{int(patch.get_height())}",
                (
                    patch.get_x()
                    + patch.get_width() / 2,
                    patch.get_height()
                ),
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            df[FEATURES].corr(),
            cmap="coolwarm",
            annot=False,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Feature Correlation Heatmap"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:

        selected_feature = st.selectbox(
            "Select Feature",
            FEATURES
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        for cls in df[TARGET].unique():

            subset = df.loc[
                df[TARGET] == cls,
                selected_feature
            ]

            ax.hist(
                subset,
                bins=30,
                alpha=0.5,
                label=cls
            )

        ax.set_xlabel(
            selected_feature
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            f"Distribution of {selected_feature} by Class"
        )

        ax.legend(fontsize=7)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab5:

        feature_info = pd.DataFrame({
            "Feature": FEATURES,
            "Min": [
                round(df[f].min(), 4)
                for f in FEATURES
            ],
            "Max": [
                round(df[f].max(), 4)
                for f in FEATURES
            ],
            "Mean": [
                round(df[f].mean(), 4)
                for f in FEATURES
            ],
            "Median": [
                round(df[f].median(), 4)
                for f in FEATURES
            ]
        })

        st.dataframe(
            feature_info,
            use_container_width=True
        )


# ==========================================================
# SINGLE PREDICTION
# ==========================================================

elif page == "🤖 Single Prediction":

    st.title("🤖 Single Bean Prediction")

    st.markdown(
        "Enter feature values to predict the bean variety. "
        "The saved StandardScaler is applied automatically "
        "before prediction."
    )

    st.divider()

    with st.expander(
        "⚙️ Input Features",
        expanded=True
    ):

        input_data = []

        col1, col2 = st.columns(2)

        for i, feature in enumerate(FEATURES):

            with col1 if i % 2 == 0 else col2:

                value = st.number_input(
                    feature,
                    value=float(
                        df[feature].median()
                    ),
                    format="%.4f"
                )

            input_data.append(value)

    selected_model = st.selectbox(
        "🤖 Select Model",
        list(models.keys())
    )

    if st.button(
        "🔍 Predict Bean Class",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        input_df = input_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if input_df.isnull().any().any():

            st.error(
                "Invalid feature value detected. "
                "Please enter valid numeric values."
            )

        else:

            scaled = scaler.transform(
                input_df
            )

            model = models[
                selected_model
            ]

            pred = model.predict(
                scaled
            )[0]

            bean = encoder.inverse_transform(
                [pred]
            )[0]

            st.markdown(
                f'<div class="prediction-box">'
                f'🌱 Predicted Bean Class: {bean}'
                f'</div>',
                unsafe_allow_html=True
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    scaled
                )[0]

                prob_df = pd.DataFrame({
                    "Bean Class": encoder.classes_,
                    "Probability": probability
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                top_confidence = (
                    prob_df.iloc[0]["Probability"]
                    * 100
                )

                st.metric(
                    "Prediction Confidence",
                    f"{top_confidence:.1f}%"
                )

                st.markdown(
                    "**Top 3 Predicted Classes**"
                )

                top3 = prob_df.head(3).copy()

                top3["Probability"] = (
                    top3["Probability"]
                    .map(lambda x: f"{x:.2%}")
                )

                st.dataframe(
                    top3,
                    use_container_width=True,
                    hide_index=True
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**All Prediction Probabilities**"
                    )

                    st.dataframe(
                        prob_df.style
                        .format({
                            "Probability": "{:.2%}"
                        })
                        .background_gradient(
                            subset=["Probability"],
                            cmap="Greens"
                        ),
                        use_container_width=True
                    )

                with col2:

                    fig, ax = plt.subplots(
                        figsize=(6, 4)
                    )

                    bar_colors = [
                        "#38ef7d"
                        if variety == bean
                        else "#667eea"
                        for variety in prob_df[
                            "Bean Class"
                        ]
                    ]

                    ax.barh(
                        prob_df["Bean Class"],
                        prob_df["Probability"],
                        color=bar_colors
                    )

                    ax.set_xlabel(
                        "Probability"
                    )

                    ax.set_title(
                        "Prediction Probability by Bean Class"
                    )

                    ax.invert_yaxis()

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)


# ==========================================================
# BATCH PREDICTION
# ==========================================================

elif page == "📁 Batch Prediction":

    st.title("📁 Batch Prediction")

    st.markdown(
        "Upload a CSV with the 16 feature columns to "
        "predict all records at once."
    )

    st.divider()

    # Sample template
    sample_df = df[FEATURES].head(3).copy()

    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=sample_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="DryBean_Sample_Template.csv",
        mime="text/csv",
        help="Use this file as a template for batch prediction."
    )

    st.caption(
        "The uploaded CSV must contain all 16 feature columns "
        "with the same column names as the template."
    )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:

        uploaded = st.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )

    with col2:

        selected_model = st.selectbox(
            "Select Model",
            list(models.keys())
        )

    if uploaded is not None:

        try:

            batch = pd.read_csv(
                uploaded
            )

            st.markdown(
                "**Uploaded Data Preview**"
            )

            st.dataframe(
                batch.head(),
                use_container_width=True
            )

            if st.button(
                "▶ Run Batch Prediction",
                use_container_width=True
            ):

                missing = [
                    feature
                    for feature in FEATURES
                    if feature not in batch.columns
                ]

                if missing:

                    st.error(
                        "Missing columns: "
                        + ", ".join(missing)
                    )

                elif len(batch) == 0:

                    st.error(
                        "The uploaded CSV contains no records."
                    )

                else:

                    batch[FEATURES] = batch[
                        FEATURES
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    batch = batch.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    invalid_columns = batch[
                        FEATURES
                    ].columns[
                        batch[FEATURES]
                        .isnull()
                        .any()
                    ].tolist()

                    if invalid_columns:

                        st.error(
                            "Invalid or missing values found in: "
                            + ", ".join(invalid_columns)
                            + ". Please correct the CSV "
                              "and upload it again."
                        )

                    else:

                        scaled = scaler.transform(
                            batch[FEATURES]
                        )

                        prediction = models[
                            selected_model
                        ].predict(
                            scaled
                        )

                        prediction = encoder.inverse_transform(
                            prediction
                        )

                        batch[
                            "Predicted_Class"
                        ] = prediction

                        st.success(
                            f"✅ Prediction complete — "
                            f"{len(batch):,} records processed."
                        )

                        st.markdown(
                            "**Prediction Summary**"
                        )

                        summary = (
                            batch[
                                "Predicted_Class"
                            ]
                            .value_counts()
                            .reset_index()
                        )

                        summary.columns = [
                            "Bean Class",
                            "Number of Records"
                        ]

                        col1, col2 = st.columns(2)

                        with col1:

                            st.dataframe(
                                summary,
                                use_container_width=True
                            )

                        with col2:

                            chart_summary = summary.sort_values(
                                "Number of Records",
                                ascending=True
                            )

                            fig, ax = plt.subplots(
                                figsize=(5.5, 4)
                            )

                            ax.barh(
                                chart_summary["Bean Class"],
                                chart_summary["Number of Records"],
                                color=sns.color_palette(
                                    "viridis",
                                    len(chart_summary)
                                )
                            )

                            ax.set_xlabel("Number of Records")
                            ax.set_title(
                                "Predicted Bean Class Distribution"
                            )

                            for i, value in enumerate(
                                chart_summary["Number of Records"]
                            ):
                                ax.text(
                                    value,
                                    i,
                                    f" {int(value)}",
                                    va="center",
                                    fontsize=9
                                )

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        st.dataframe(
                            batch,
                            use_container_width=True
                        )

                        st.download_button(
                            "⬇ Download Results",
                            data=batch.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name=(
                                "DryBean_Predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:

            st.error(
                f"Unable to process the uploaded CSV: {e}"
            )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

elif page == "📈 Model Comparison":

    st.title("📈 Model Performance Comparison")
    st.divider()

    try:

        comparison = get_model_comparison()

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
            "ROC-AUC"
        ]

        comparison = comparison.dropna(
            subset=["Model", "Accuracy"]
        ).copy()

        if comparison.empty:

            st.error(
                "No valid model comparison results found."
            )
            st.stop()

        # Best accuracy
        best_accuracy = comparison[
            "Accuracy"
        ].max()

        best_accuracy_models = comparison.loc[
            comparison["Accuracy"] == best_accuracy,
            "Model"
        ].tolist()

        # Best F1
        best_f1 = comparison[
            "F1 Score"
        ].max()

        best_f1_models = comparison.loc[
            comparison["F1 Score"] == best_f1,
            "Model"
        ].tolist()

        # Best ROC-AUC
        best_auc = comparison[
            "ROC-AUC"
        ].max()

        best_auc_models = comparison.loc[
            comparison["ROC-AUC"] == best_auc,
            "Model"
        ].tolist()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🏆 Best Accuracy",
            f"{best_accuracy:.4f}"
        )

        c2.metric(
            "📊 Best F1 Score",
            f"{best_f1:.4f}"
        )

        c3.metric(
            "📈 Best ROC-AUC",
            f"{best_auc:.4f}"
        )

        st.markdown(
            f"**Best Accuracy Model(s):** "
            f"{', '.join(best_accuracy_models)}"
        )

        st.markdown(
            f"**Best F1 Model(s):** "
            f"{', '.join(best_f1_models)}"
        )

        st.markdown(
            f"**Best ROC-AUC Model(s):** "
            f"{', '.join(best_auc_models)}"
        )

        st.divider()

        st.caption(
            "All reported metrics are taken from the saved model-evaluation "
            "results. The confusion matrices are generated from the held-out "
            "test data using the same saved StandardScaler and label encoder."
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Metrics Table",
            "📊 Bar Chart",
            "🕸 Radar Chart",
            "🔲 Confusion Matrix"
        ])

        with tab1:

            display_comparison = comparison[
                ["Model"] + metrics
            ].copy()

            st.dataframe(
                display_comparison.style
                .background_gradient(
                    subset=metrics,
                    cmap="Greens"
                )
                .format({
                    metric: "{:.4f}"
                    for metric in metrics
                }),
                use_container_width=True
            )

        with tab2:

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            comparison.set_index(
                "Model"
            )[metrics].plot(
                kind="bar",
                ax=ax,
                colormap="viridis"
            )

            ax.set_ylabel(
                "Score"
            )

            ax.set_ylim(
                0.0,
                1.0
            )

            ax.legend(
                loc="lower right",
                fontsize=8
            )

            plt.xticks(
                rotation=20
            )

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        with tab3:

            radar_metrics = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ]

            N = len(
                radar_metrics
            )

            angles = [
                n / float(N) * 2 * np.pi
                for n in range(N)
            ]

            angles += angles[:1]

            fig, ax = plt.subplots(
                figsize=(7, 7),
                subplot_kw=dict(
                    polar=True
                )
            )

            colors = sns.color_palette(
                "viridis",
                len(comparison)
            )

            for i, (_, row) in enumerate(
                comparison.iterrows()
            ):

                values = [
                    row[metric]
                    for metric in radar_metrics
                ]

                values += values[:1]

                ax.plot(
                    angles,
                    values,
                    color=colors[i],
                    linewidth=2,
                    label=row["Model"]
                )

                ax.fill(
                    angles,
                    values,
                    color=colors[i],
                    alpha=0.1
                )

            ax.set_xticks(
                angles[:-1]
            )

            ax.set_xticklabels(
                radar_metrics,
                fontsize=10
            )

            ax.set_ylim(
                0.85,
                1.0
            )

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.3, 1.1),
                fontsize=9
            )

            ax.set_title(
                "Model Radar Comparison",
                pad=20
            )

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        with tab4:

            st.markdown(
                "**Confusion Matrix by Model**"
            )

            st.caption(
                "Select a model to view its confusion matrix and "
                "classification report on the held-out test set."
            )

            selected_cm_model = st.selectbox(
                "Select Model",
                list(models.keys()),
                key="cm_model"
            )

            try:
                test_data = pd.read_csv(
                    "output/test_data.csv"
                )

                test_data = test_data.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                # Convert all feature columns to numeric values.
                test_data[FEATURES] = test_data[
                    FEATURES
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Validate the target column.
                if TARGET not in test_data.columns:
                    st.error(
                        f"The test-data file does not contain the "
                        f"required target column '{TARGET}'."
                    )
                else:
                    test_data = test_data.dropna(
                        subset=FEATURES + [TARGET]
                    ).copy()

                    if test_data.empty:
                        st.error(
                            "No valid test records are available "
                            "for the confusion matrix."
                        )
                    else:
                        X_test = test_data[FEATURES]

                        # The saved models were trained on scaled features.
                        X_test_scaled = scaler.transform(X_test)

                        # Convert the actual target into the same encoded
                        # label space used by the saved models.
                        y_raw = test_data[TARGET]

                        if pd.api.types.is_numeric_dtype(y_raw):
                            y_test = y_raw.astype(int).to_numpy()
                        else:
                            y_test = encoder.transform(
                                y_raw.astype(str)
                            )

                        model = models[selected_cm_model]

                        # IMPORTANT: predict using the scaled test data.
                        y_pred_raw = model.predict(
                            X_test_scaled
                        )

                        # Robustly handle either encoded or string predictions.
                        if pd.api.types.is_numeric_dtype(
                            pd.Series(y_pred_raw)
                        ):
                            y_pred = np.asarray(
                                y_pred_raw,
                                dtype=int
                            )
                        else:
                            y_pred = encoder.transform(
                                pd.Series(
                                    y_pred_raw
                                ).astype(str)
                            )

                        labels = np.arange(
                            len(encoder.classes_)
                        )

                        cm = confusion_matrix(
                            y_test,
                            y_pred,
                            labels=labels
                        )

                        fig_cm, ax_cm = plt.subplots(
                            figsize=(9, 7)
                        )

                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            linewidths=0.5,
                            linecolor="white",
                            xticklabels=encoder.classes_,
                            yticklabels=encoder.classes_,
                            ax=ax_cm
                        )

                        ax_cm.set_xlabel(
                            "Predicted Bean Class"
                        )

                        ax_cm.set_ylabel(
                            "Actual Bean Class"
                        )

                        ax_cm.set_title(
                            f"Confusion Matrix — {selected_cm_model}"
                        )

                        plt.tight_layout()

                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                        # Classification report using exactly the same
                        # predictions as the confusion matrix.
                        report = classification_report(
                            y_test,
                            y_pred,
                            labels=labels,
                            target_names=encoder.classes_,
                            output_dict=True,
                            zero_division=0
                        )

                        report_df = pd.DataFrame(
                            report
                        ).transpose()

                        st.markdown(
                            "**Classification Report**"
                        )

                        st.dataframe(
                            report_df.style
                            .format("{:.4f}")
                            .background_gradient(
                                cmap="Greens"
                            ),
                            use_container_width=True
                        )

            except Exception as cm_error:
                st.error(
                    "Unable to generate the confusion matrix: "
                    f"{cm_error}"
                )

    except Exception as e:

        st.error(
            f"Error loading model comparison: {e}"
        )


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif page == "🌳 Feature Importance":

    st.title("🌳 Feature Importance")
    st.divider()

    rf = models["Random Forest"]

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    importance["Rank"] = range(
        1,
        len(importance) + 1
    )

    # Top N selector
    top_n = st.slider(
        "Select number of features to display",
        min_value=5,
        max_value=len(FEATURES),
        value=10,
        step=1
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "**Feature Importance Scores**"
        )

        st.dataframe(
            importance[
                [
                    "Rank",
                    "Feature",
                    "Importance"
                ]
            ].head(top_n)
            .style
            .format({
                "Importance": "{:.4f}"
            })
            .background_gradient(
                subset=["Importance"],
                cmap="Greens"
            ),
            use_container_width=True
        )

    with col2:

        top_importance = importance.head(
            top_n
        ).sort_values(
            "Importance",
            ascending=True
        )

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        colors = sns.color_palette(
            "viridis",
            len(top_importance)
        )

        ax.barh(
            top_importance["Feature"],
            top_importance["Importance"],
            color=colors
        )

        ax.set_xlabel(
            "Importance Score"
        )

        ax.set_title(
            f"Top {top_n} Random Forest Features"
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.info(
        "Feature importance is based on the trained Random Forest model. "
        "Higher scores indicate features that contributed more to the "
        "model's impurity-based split decisions; they should not be "
        "interpreted as causal effects."
    )

    importance["Cumulative"] = (
        importance["Importance"].cumsum()
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 4)
    )

    ax2.plot(
        importance["Feature"],
        importance["Cumulative"],
        marker="o",
        color="#667eea"
    )

    ax2.axhline(
        y=0.8,
        color="red",
        linestyle="--",
        label="80% threshold"
    )

    ax2.set_ylabel(
        "Cumulative Importance"
    )

    ax2.set_title(
        "Cumulative Feature Importance"
    )

    ax2.legend()

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)

    top3 = importance[
        "Feature"
    ].head(3).tolist()

    st.info(
        "Top 3 most important features: "
        f"**{', '.join(top3)}**"
    )


# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
### 🌱 Dry Bean Classification

Developed as part of **BITS Pilani WILP – AIMLCZG565 Machine Learning, Assignment 2**.

**Dataset:** Dry Bean Dataset (UCI ML Repository)

- 13,611 records
- 16 features
- 7 classes

**Models:**
Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest

**Evaluation Metrics:**
Accuracy, Precision, Recall, F1 Score, MCC, and ROC-AUC
""")

    with col2:

        st.markdown("""
### 👩‍🎓 Student Details

| Field | Details |
|---|---|
| Name | Mansi Jain |
| ID | 2025AC05151 |
| Course | AIMLCZG565 |
| Programme | M.Tech AI & ML |
| University | BITS Pilani WILP |
""")

    st.divider()

    st.markdown(
        '<div class="section-header">🔄 Methodology</div>',
        unsafe_allow_html=True
    )

    st.info("""
    **Dataset**
    → **EDA**
    → **Preprocessing**
    → **Train/Test Split**
    → **Feature Scaling**
    → **Model Training**
    → **Hyperparameter Tuning**
    → **Evaluation**
    → **Model Comparison**
    → **Feature Importance**
    → **Streamlit Deployment**
    """)

    st.markdown(
        '<div class="section-header">🛠 Technologies</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.metric("Python", "ML")
    tech2.metric("Scikit-learn", "Models")
    tech3.metric("Streamlit", "Deployment")
    tech4.metric("Joblib", "Model Saving")

    st.divider()

    st.markdown(
        '<div class="section-header">💻 Project Code</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    The complete implementation is maintained in the GitHub repository,
    including the training notebooks, Streamlit application,
    saved model files and generated output files.
    """)

    if st.session_state.user_name:

        st.markdown(
            '<div class="section-header">👤 Session Information</div>',
            unsafe_allow_html=True
        )

        session_info = pd.DataFrame({
            "Information": [
                "User",
                "Session Date",
                "Session Time",
                "Application",
                "Course"
            ],
            "Value": [
                st.session_state.user_name,
                current_date,
                current_time,
                "Dry Bean Classification",
                "AIMLCZG565 – Machine Learning"
            ]
        })

        st.dataframe(
            session_info,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

    c2.metric(
        "ML Models",
        len(models)
    )

    c3.metric(
        "Classes",
        df[TARGET].nunique()
    )

    c4.metric(
        "Features",
        len(FEATURES)
    )

    st.success(
        "✔ Developed using Python · Streamlit · Scikit-learn"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<center>
<small>
<b>BITS Pilani Work Integrated Learning Programme</b>
&nbsp;|&nbsp;
AIMLCZG565 – Machine Learning
&nbsp;|&nbsp;
Assignment 2
&nbsp;|&nbsp;
Dry Bean Classification
&nbsp;|&nbsp;
© 2026 Mansi Jain
</small>
</center>
""", unsafe_allow_html=True)
