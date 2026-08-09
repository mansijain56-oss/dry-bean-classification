import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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

    .section-header {
        border-left: 5px solid #667eea;
        padding-left: 12px;
        margin: 20px 0 10px 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    "🎓 **BITS Pilani WILP**",
    unsafe_allow_html=True
)

st.sidebar.title("🌱 Dry Bean Classifier")
st.sidebar.caption("AIMLCZG565 – Machine Learning | Assignment 2")
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
    """
    Load the dataset from the repository itself.
    Keeping the file local avoids an unnecessary external dependency.
    """
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


# Load application resources
df = load_dataset()
models, scaler, encoder = load_models()


# ==========================================================
# BASIC VALIDATION
# ==========================================================

missing_features = [feature for feature in FEATURES if feature not in df.columns]

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
st.sidebar.markdown(f"- Classes: **{df[TARGET].nunique()}**")
st.sidebar.markdown(f"- Models: **{len(models)}**")


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, X_test, y_test):
    """
    Calculate standard classification metrics.
    Kept as a reusable helper for the project.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
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
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

        classes = np.arange(len(encoder.classes_))
        y_bin = label_binarize(y_test, classes=classes)

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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">🎯 Objectives</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        - Explore and analyze the Dry Bean Dataset
        - Train and evaluate five ML classifiers
        - Compare model performance across six metrics
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
        colors = sns.color_palette("viridis", len(counts))

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

    # Dynamic best-model message with tie handling
    try:
        comparison = pd.read_csv(
            "output/model_comparison.csv"
        )

        if "Accuracy" in comparison.columns and "Model" in comparison.columns:

            comparison["Accuracy"] = pd.to_numeric(
                comparison["Accuracy"],
                errors="coerce"
            )

            comparison = comparison.dropna(
                subset=["Accuracy"]
            )

            if not comparison.empty:
                best_accuracy = comparison["Accuracy"].max()

                best_models = comparison.loc[
                    comparison["Accuracy"] == best_accuracy,
                    "Model"
                ].tolist()

                if len(best_models) == 1:
                    st.success(
                        f"🏆 {best_models[0]} achieved the highest "
                        f"accuracy ({best_accuracy:.4f}) among all models."
                    )
                else:
                    st.success(
                        f"🏆 {', '.join(best_models)} achieved the highest "
                        f"accuracy ({best_accuracy:.4f}) among all models."
                    )
            else:
                st.info(
                    "Model comparison results are available on the "
                    "Model Comparison page."
                )

    except Exception:
        st.info(
            "Model comparison results are available on the "
            "Model Comparison page."
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
    c4.metric("Classes", df[TARGET].nunique())

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

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.countplot(
            data=df,
            x=TARGET,
            order=df[TARGET].value_counts().index,
            palette="viridis",
            ax=ax
        )

        ax.set_title("Bean Class Distribution")
        ax.tick_params(axis="x", rotation=30)

        for patch in ax.patches:
            ax.annotate(
                f"{int(patch.get_height())}",
                (
                    patch.get_x() + patch.get_width() / 2,
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

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            df[FEATURES].corr(),
            cmap="coolwarm",
            annot=False,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Feature Correlation Heatmap")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab4:

        selected_feature = st.selectbox(
            "Select Feature",
            FEATURES
        )

        fig, ax = plt.subplots(figsize=(9, 4))

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

        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
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
                round(df[feature].min(), 4)
                for feature in FEATURES
            ],
            "Max": [
                round(df[feature].max(), 4)
                for feature in FEATURES
            ],
            "Mean": [
                round(df[feature].mean(), 4)
                for feature in FEATURES
            ],
            "Median": [
                round(df[feature].median(), 4)
                for feature in FEATURES
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
        "Enter feature values to predict the bean variety."
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
                    value=float(df[feature].median()),
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

        # Ensure valid numeric values
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

            scaled = scaler.transform(input_df)

            pred = models[selected_model].predict(
                scaled
            )[0]

            bean = encoder.inverse_transform(
                [pred]
            )[0]

            st.markdown(
                f'<div class="prediction-box">'
                f'🌱 Predicted Bean: {bean}'
                f'</div>',
                unsafe_allow_html=True
            )

            if hasattr(
                models[selected_model],
                "predict_proba"
            ):

                probability = models[
                    selected_model
                ].predict_proba(scaled)[0]

                top_conf = probability.max() * 100

                st.metric(
                    "Top Confidence",
                    f"{top_conf:.1f}%"
                )

                prob_df = pd.DataFrame({
                    "Bean Variety": encoder.classes_,
                    "Probability": probability
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Prediction Probabilities**"
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
                            "Bean Variety"
                        ]
                    ]

                    ax.barh(
                        prob_df["Bean Variety"],
                        prob_df["Probability"],
                        color=bar_colors
                    )

                    ax.set_xlabel("Probability")
                    ax.set_title(
                        "Confidence per Class"
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

            batch = pd.read_csv(uploaded)

            st.markdown("**Uploaded Data Preview**")

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

                else:

                    # Convert feature columns to numeric
                    batch[FEATURES] = batch[
                        FEATURES
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    # Replace infinite values
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

                    elif len(batch) == 0:

                        st.error(
                            "The uploaded CSV contains no records."
                        )

                    else:

                        scaled = scaler.transform(
                            batch[FEATURES]
                        )

                        prediction = models[
                            selected_model
                        ].predict(scaled)

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
                            "Bean Variety",
                            "Count"
                        ]

                        col1, col2 = st.columns(2)

                        with col1:

                            st.dataframe(
                                summary,
                                use_container_width=True
                            )

                        with col2:

                            fig, ax = plt.subplots(
                                figsize=(5, 4)
                            )

                            pie_colors = sns.color_palette(
                                "viridis",
                                len(summary)
                            )

                            ax.pie(
                                summary["Count"],
                                labels=summary[
                                    "Bean Variety"
                                ],
                                autopct="%1.1f%%",
                                colors=pie_colors
                            )

                            ax.set_title(
                                "Predicted Class Distribution"
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

        comparison = pd.read_csv(
            "output/model_comparison.csv"
        )

        required_metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
            "ROC-AUC"
        ]

        missing_metrics = [
            metric
            for metric in required_metrics
            if metric not in comparison.columns
        ]

        if "Model" not in comparison.columns:

            st.error(
                "The model comparison file does not contain "
                "a 'Model' column."
            )
            st.stop()

        if missing_metrics:

            st.error(
                "Missing metric columns: "
                + ", ".join(missing_metrics)
            )
            st.stop()

        metrics = required_metrics

        # Ensure metric columns are numeric
        for metric in metrics:

            comparison[metric] = pd.to_numeric(
                comparison[metric],
                errors="coerce"
            )

        # Remove invalid rows only if necessary
        comparison = comparison.dropna(
            subset=["Model", "Accuracy"]
        ).copy()

        if comparison.empty:

            st.error(
                "No valid model comparison results found."
            )
            st.stop()

        # Top model cards
        best_accuracy = comparison[
            "Accuracy"
        ].max()

        best_models = comparison.loc[
            comparison["Accuracy"] == best_accuracy,
            "Model"
        ].tolist()

        best_f1 = comparison[
            "F1 Score"
        ].max()

        best_f1_models = comparison.loc[
            comparison["F1 Score"] == best_f1,
            "Model"
        ].tolist()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🏆 Best Accuracy Model",
            ", ".join(best_models)
        )

        c2.metric(
            "🎯 Best Accuracy",
            f"{best_accuracy:.4f}"
        )

        c3.metric(
            "📊 Best F1 Score",
            f"{best_f1:.4f}"
        )

        st.caption(
            "Best F1 Score: "
            + ", ".join(best_f1_models)
        )

        st.divider()

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

            ax.set_ylabel("Score")
            ax.set_ylim(0.85, 1.0)
            ax.legend(
                loc="lower right",
                fontsize=8
            )

            plt.xticks(rotation=20)
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

            N = len(radar_metrics)

            angles = [
                n / float(N) * 2 * np.pi
                for n in range(N)
            ]

            angles += angles[:1]

            fig, ax = plt.subplots(
                figsize=(7, 7),
                subplot_kw=dict(polar=True)
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
                "**Confusion Matrix per Model**"
            )

            selected_cm_model = st.selectbox(
                "Select Model for Confusion Matrix",
                list(models.keys()),
                key="cm_model"
            )

            test_data = pd.read_csv(
                "output/test_data.csv"
            )

            # Clean invalid values safely
            test_data = test_data.replace(
                [np.inf, -np.inf],
                np.nan
            )

            test_data = test_data.dropna(
                subset=FEATURES + [TARGET]
            ).copy()

            if test_data.empty:

                st.error(
                    "No valid test records are available "
                    "for the confusion matrix."
                )

            else:

                # Ensure numeric test values
                test_data[FEATURES] = test_data[
                    FEATURES
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                test_data[TARGET] = pd.to_numeric(
                    test_data[TARGET],
                    errors="coerce"
                )

                test_data = test_data.dropna(
                    subset=FEATURES + [TARGET]
                )

                X_test = test_data[
                    FEATURES
                ]

                y_test = test_data[
                    TARGET
                ].astype(int)

                y_pred = models[
                    selected_cm_model
                ].predict(X_test)

                # Explicit labels guarantee a stable
                # 7 x 7 confusion matrix
                labels = np.arange(
                    len(encoder.classes_)
                )

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=labels
                )

                fig_cm, ax_cm = plt.subplots(
                    figsize=(8, 6)
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=encoder.classes_,
                    yticklabels=encoder.classes_,
                    ax=ax_cm
                )

                ax_cm.set_xlabel(
                    "Predicted"
                )

                ax_cm.set_ylabel(
                    "Actual"
                )

                ax_cm.set_title(
                    f"Confusion Matrix - "
                    f"{selected_cm_model}"
                )

                plt.tight_layout()

                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # Classification report
                st.markdown(
                    "**Classification Report**"
                )

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

                st.dataframe(
                    report_df.style
                    .format("{:.4f}")
                    .background_gradient(
                        cmap="Greens"
                    ),
                    use_container_width=True
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

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "**Feature Importance Scores**"
        )

        st.dataframe(
            importance[
                ["Rank", "Feature", "Importance"]
            ]
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

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        colors = sns.color_palette(
            "viridis",
            len(importance)
        )

        ax.barh(
            importance["Feature"],
            importance["Importance"],
            color=colors
        )

        ax.set_xlabel(
            "Importance Score"
        )

        ax.set_title(
            "Random Forest Feature Importance"
        )

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Cumulative importance
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

Developed as part of **BITS Pilani WILP – AIMLCZG565 Machine Learning Assignment 2**.

**Dataset:** Dry Bean Dataset (UCI ML Repository)

- 13,611 records
- 16 features
- 7 classes

**Models:**
Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest

**Metrics:**
Accuracy, Precision, Recall, F1, MCC, ROC-AUC
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
