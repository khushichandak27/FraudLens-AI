import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
from PIL import Image

# ---------- Page Config ----------
logo_icon = Image.open("assets/logo.png")

st.set_page_config(
    page_title="FraudLens AI",
    page_icon=logo_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom Theme (CSS) ----------
st.markdown("""
    <style>
    .stApp {
        background: #f5f7fa;
        color: #0A1A2F;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d0d5dd;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 34px !important;
        font-weight: 700;
        color: #0A1A2F !important;
    }
    section[data-testid="stSidebar"] p {
        font-size: 16px !important;
        color: #0A1A2F !important;
    }
    section[data-testid="stSidebar"] label p {
        font-size: 19px !important;
        font-weight: 500;
        color: #0A1A2F !important;
    }
    section[data-testid="stSidebar"] input[type="radio"] {
        accent-color: #1B3A6B !important;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 10px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #0A1A2F !important;
    }
    p, span, div {
        color: #0A1A2F;
    }
    .stButton>button {
        background-color: #1B3A6B;
        color: #ffffff;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    [data-testid="stMetricValue"] {
        color: #0A1A2F !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4a5568 !important;
    }
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #0A1A2F !important;
        border: 1px solid #d0d5dd !important;
    }
    .stNumberInput button {
        background-color: #f0f2f5 !important;
        color: #0A1A2F !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #ffffff !important;
        color: #0A1A2F !important;
        border: 1px solid #d0d5dd !important;
    }
    [data-testid="stExpander"] summary p {
        color: #0A1A2F !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
    }
    [data-testid="stDataFrame"] div {
        color: #0A1A2F !important;
    }
    .stTable table {
        background-color: #ffffff !important;
    }
    .stTable th {
        background-color: #ffffff !important;
        color: #0A1A2F !important;
    }
    .stTable td {
        background-color: #ffffff !important;
        color: #0A1A2F !important;
    }
    .stTable tbody th {
        background-color: #ffffff !important;
        color: #0A1A2F !important;
    }
    .stButton>button p {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("assets/logo.png", width=150)
    st.markdown("## FraudLens AI")
    st.caption("Credit Card Fraud Detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Dataset Overview", "Transaction Analyzer", "Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Built with Random Forest · SHAP · Streamlit")

# ---------- Cached Data & Model Loading ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard_sample.csv")
    seconds_in_day = df["Time"] % (24 * 3600)
    df["Hour"] = seconds_in_day // 3600
    df["Minute"] = (seconds_in_day % 3600) // 60

    amount_scaler = joblib.load("models/amount_scaler.pkl")
    time_scaler = joblib.load("models/time_scaler.pkl")

    df["scaled_amount"] = amount_scaler.transform(df[["Amount"]])
    df["scaled_time"] = time_scaler.transform(df[["Time"]])

    return df

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

df = load_data()
model = load_model()
explainer = load_explainer(model)

# ---------- Page: Dataset Overview ----------
if page == "Dataset Overview":
    st.title(" FraudLens AI — Dataset Overview")
    st.write("A snapshot of the credit card transaction dataset used to train the fraud detection model.")

    total_txn = len(df)
    fraud_txn = int(df["Class"].sum())
    fraud_rate = fraud_txn / total_txn * 100
    avg_amount = df["Amount"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_txn:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_txn:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Transaction Amount", f"${avg_amount:.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Genuine vs Fraud")
        class_counts = df["Class"].value_counts().rename({0: "Genuine", 1: "Fraud"})
        fig_donut = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            hole=0.55,
            color=class_counts.index,
            color_discrete_map={"Genuine": "#1B3A6B", "Fraud": "#BFBFBF"}
        )
        fig_donut.update_traces(marker=dict(line=dict(color="#ffffff", width=2)))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#0A1A2F",
            legend=dict(font=dict(color="#0A1A2F"))
        )
        st.plotly_chart(fig_donut, use_container_width=True, theme=None)

    with c2:
        st.subheader("Fraud Count by Hour of Day")
        hourly_fraud = df[df["Class"] == 1].groupby("Hour").size().reset_index(name="Fraud Count")
        fig_bar = px.bar(
            hourly_fraud, x="Hour", y="Fraud Count",
            color="Fraud Count",
            color_continuous_scale=["#BFBFBF", "#5B7FA6", "#1B3A6B"]
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#0A1A2F",
            xaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F"),
            yaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)
        
# ---------- Page: Transaction Analyzer ----------
elif page == "Transaction Analyzer":
    st.title(" Transaction Analyzer")
    st.write("Enter a transaction ID (row index from the test set) to check whether it's genuine or fraudulent.")

    max_idx = len(df) - 1

    if "txn_id" not in st.session_state:
        st.session_state.txn_id = 0

    def pick_random_fraud():
        st.session_state.txn_id = int(df[df["Class"] == 1].sample(1).index[0])

    txn_id = st.number_input(
        "Transaction ID",
        min_value=0,
        max_value=max_idx,
        step=1,
        key="txn_id"
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        analyze_btn = st.button("Analyze Transaction")
    with col_b:
        st.button("Try a Random Fraud Case", on_click=pick_random_fraud)

    if analyze_btn:
        row = df.loc[[st.session_state.txn_id]]
        features = row.drop(columns=["Class", "Hour", "Minute", "Amount", "Time"])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(f"🚨 FRAUD DETECTED — Confidence: {probability*100:.2f}%")
        else:
            st.success(f"✅ GENUINE TRANSACTION — Fraud Probability: {probability*100:.2f}%")

        st.progress(float(probability))

        c1, c2, c3 = st.columns(3)
        c1.metric("Transaction Amount", f"${row['Amount'].values[0]:.2f}")
        c2.metric("Time of Day", f"{int(row['Hour'].values[0]):02d}:{int(row['Minute'].values[0]):02d}")
        c3.metric("Actual Label", "Fraud" if row["Class"].values[0] == 1 else "Genuine")

        with st.expander("View Raw Feature Values"):
            st.table(features.T.rename(columns={st.session_state.txn_id: "Value"}))

        st.markdown("### Why This Prediction?")
        shap_values = explainer.shap_values(features)

        shap.force_plot(
            explainer.expected_value[1],
            shap_values[0, :, 1],
            features,
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        fig.patch.set_alpha(0)

        for ax in fig.axes:
            ax.patch.set_alpha(0)
            for text in ax.texts:
                text.set_color("#0A1A2F")
                text.set_bbox(dict(facecolor='none', edgecolor='none'))
            ax.tick_params(colors="#0A1A2F")
            for spine in ax.spines.values():
                spine.set_color("#0A1A2F")

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches='tight')
        st.image(buf)
        plt.close(fig)

elif page == "Model Performance":
    st.title(" Model Performance")
    st.write("Evaluation of the Random Forest model on the held-out test set.")

    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = joblib.load("data/processed_data.pkl")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, roc_auc_score,
        precision_recall_curve, average_precision_score
    )

    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Key Metrics (Fraud Class)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{report['1']['precision']:.3f}")
    c2.metric("Recall", f"{report['1']['recall']:.3f}")
    c3.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
    c4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Genuine", "Fraud"],
            y=["Genuine", "Fraud"],
            color_continuous_scale=["#8FC1C0", "#0A1A2F"],
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#0A1A2F",
            coloraxis_showscale=False
        )
        fig_cm.update_xaxes(color="#0A1A2F")
        fig_cm.update_yaxes(color="#0A1A2F")
        st.plotly_chart(fig_cm, use_container_width=True, theme=None)

    with c2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color="#1B3A6B"), name="Random Forest"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color="#BFBFBF", dash="dash"), name="Random Guess"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#0A1A2F",
            legend_font_color="#0A1A2F",
            xaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F"),
            yaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F")
        )
        st.plotly_chart(fig_roc, use_container_width=True, theme=None)

    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color="#1B3A6B"), name=f"AP = {ap_score:.4f}"))
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0A1A2F",
        legend_font_color="#0A1A2F",
        xaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F"),
        yaxis=dict(gridcolor="#e2e5ea", color="#0A1A2F")
    )
    st.plotly_chart(fig_pr, use_container_width=True, theme=None)

    st.markdown("---")
    st.subheader("Global Feature Importance (SHAP)")

    @st.cache_data
    def compute_shap_sample(_model, _X_test):
        X_sample = _X_test.sample(300, random_state=42)
        shap_values_sample = explainer.shap_values(X_sample)
        return X_sample, shap_values_sample

    with st.spinner("Computing SHAP values..."):
        X_sample, shap_values_sample = compute_shap_sample(model, X_test)

    shap.summary_plot(shap_values_sample[:,:,1], X_sample, show=False, plot_type="bar")
    fig_shap = plt.gcf()
    ax = plt.gca()

    for bar in ax.patches:
        bar.set_color("#1B3A6B")

    fig_shap.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.tick_params(colors="#0A1A2F")
    ax.xaxis.label.set_color("#0A1A2F")
    ax.yaxis.label.set_color("#0A1A2F")
    for label in ax.get_yticklabels():
        label.set_color("#0A1A2F")
    for spine in ax.spines.values():
        spine.set_color("#0A1A2F")
    st.pyplot(fig_shap)
    plt.close(fig_shap)
