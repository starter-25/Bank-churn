import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Customer Churn Analysis & Prediction")
st.markdown("Upload your Bank dataset to explore, preprocess, and train ML models.")

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset", type=["csv"],
    help="Upload your Bank_Dataset CSV file"
)

# ─── Helper: load + cache data ──────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def preprocess(raw: pd.DataFrame):
    df = raw.copy()
    # Drop irrelevant columns if present
    drop_cols = [c for c in ['id', 'CustomerId', 'Surname'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Outlier trimming (IQR)
    for col in ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    df.reset_index(drop=True, inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in ['Gender', 'Geography']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df

# ─── Main Flow ──────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

raw_df = load_data(uploaded_file)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Raw Data", "📊 EDA", "🔧 Preprocessing", "🤖 Model Training"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Raw Data
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", raw_df.shape[0])
    col2.metric("Columns", raw_df.shape[1])
    col3.metric("Missing Values", int(raw_df.isna().sum().sum()))

    st.dataframe(raw_df.head(50), use_container_width=True)

    st.subheader("Data Types & Describe")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**dtypes**")
        st.dataframe(raw_df.dtypes.rename("dtype").reset_index(), use_container_width=True)
    with c2:
        st.write("**describe()**")
        st.dataframe(raw_df.describe(), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    # Correlation heatmap
    st.markdown("#### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(raw_df.corr(numeric_only=True), annot=True, cmap="Dark2", ax=ax)
    ax.set_title("Correlation of Columns")
    st.pyplot(fig)
    plt.close(fig)

    # Churn distribution
    if "Exited" in raw_df.columns:
        st.markdown("#### 📈 Churn Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(x="Exited", data=raw_df, ax=axes[0])
        axes[0].set_title("Overall Churn")

        if "Geography" in raw_df.columns:
            sns.countplot(x="Geography", hue="Exited", data=raw_df, ax=axes[1])
            axes[1].set_title("Location-Based Churn")
        st.pyplot(fig)
        plt.close(fig)

        if "IsActiveMember" in raw_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="IsActiveMember", hue="Exited", data=raw_df, ax=ax)
            ax.set_title("Active Member vs Churn")
            st.pyplot(fig)
            plt.close(fig)

    # Histogram
    st.markdown("#### 📊 Feature Histograms")
    fig = plt.figure(figsize=(14, 10))
    raw_df.hist(ax=fig.gca(), figsize=(14, 10))
    plt.suptitle("Histograms")
    st.pyplot(fig)
    plt.close(fig)

    # Boxplot
    st.markdown("#### 📦 Boxplot (Numeric Features)")
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select column for boxplot", numeric_cols)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=raw_df, x=selected_col, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – Preprocessing
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Data Preprocessing")

    with st.spinner("Preprocessing data…"):
        df_clean = preprocess(raw_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Rows", raw_df.shape[0])
    col2.metric("Cleaned Rows", df_clean.shape[0])
    col3.metric("Rows Removed", raw_df.shape[0] - df_clean.shape[0])

    st.write("**Cleaned Dataset (first 50 rows)**")
    st.dataframe(df_clean.head(50), use_container_width=True)

    st.write("**Null values after cleaning:**", df_clean.isna().sum().sum())

    if "Exited" in df_clean.columns:
        st.write("**Class balance:**")
        st.bar_chart(df_clean["Exited"].value_counts())

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – Model Training
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Training & Evaluation")

    if "Exited" not in raw_df.columns:
        st.error("Column 'Exited' not found in dataset.")
        st.stop()

    df_clean = preprocess(raw_df)

    X = df_clean.drop("Exited", axis=1)
    y = df_clean["Exited"]

    # Sidebar training options
    test_size   = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    # Model selector
    model_options = {
        "Logistic Regression":       LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors":       KNeighborsClassifier(),
        "Decision Tree":             DecisionTreeClassifier(),
        "SVM":                       SVC(probability=True),
        "Naive Bayes":               GaussianNB(),
        "Random Forest":             RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "AdaBoost":                  AdaBoostClassifier(),
        "Gradient Boosting":         GradientBoostingClassifier(),
        "XGBoost":                   XGBClassifier(eval_metric="logloss", verbosity=0),
    }
    selected_models = st.multiselect(
        "Select Models to Train",
        list(model_options.keys()),
        default=["Logistic Regression", "Random Forest", "XGBoost"],
    )

    if st.button("🚀 Train Selected Models"):
        if not selected_models:
            st.warning("Please select at least one model.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_seed)
        )
        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_train)
        X_test   = scaler.transform(X_test)

        results = []
        for name in selected_models:
            model = model_options[name]
            with st.spinner(f"Training {name}…"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else None
                )
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
                results.append({"Model": name, "Accuracy": round(acc, 4),
                                "ROC-AUC": round(auc, 4) if auc else "N/A"})

        # ── Results Table ────────────────────────────────────────────────────────
        st.markdown("### 📊 Results Summary")
        results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
        st.dataframe(results_df, use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(results_df["Model"], results_df["Accuracy"], color="steelblue")
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close(fig)

        # ── Detailed Report per Model ────────────────────────────────────────────
        st.markdown("### 🔍 Detailed Reports")
        for name in selected_models:
            with st.expander(f"📋 {name}"):
                model = model_options[name]
                model.fit(X_train, y_train)          # already fitted; harmless re-fit
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else None
                )

                st.text(classification_report(y_test, y_pred))

                c1, c2 = st.columns(2)

                # Confusion matrix
                with c1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm = confusion_matrix(y_test, y_pred)
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                    plt.close(fig)

                # ROC curve
                with c2:
                    if y_prob is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        auc_score   = roc_auc_score(y_test, y_prob)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
                        ax.plot([0, 1], [0, 1], "k--")
                        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                        ax.set_title("ROC Curve")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("ROC curve not available for this model.")
