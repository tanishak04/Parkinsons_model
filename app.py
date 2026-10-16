"""
Parkinson's Disease Speech Diagnostic AI Dashboard
Streamlit v1.66+ compatible - crash-proof build
"""
import os
import sys
import warnings
import traceback

warnings.filterwarnings("ignore")
os.environ["XGBOOST_USE_CUDA"] = "0"  # Prevent any accidental CUDA init

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---- Page Config (must be first st call) ----
st.set_page_config(
    page_title="Parkinson's Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- Safe loaders with error handling ----
@st.cache_resource
def load_model_bundle():
    try:
        model_path = os.path.join(BASE_DIR, "models", "rf_model.joblib")
        if not os.path.exists(model_path):
            return None, f"Model file not found at: {model_path}"
        data = joblib.load(model_path)
        # Force CPU inference - this is the critical fix for OOM/crash
        model = data["model"]
        if hasattr(model, "set_params"):
            model.set_params(device="cpu")
        return data, None
    except Exception as e:
        return None, f"Failed to load model: {traceback.format_exc()}"


@st.cache_data
def load_raw_dataset():
    try:
        csv_path = os.path.join(BASE_DIR, "parkinsons.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception:
        pass
    return None


@st.cache_data
def run_prediction(_model, feature_values, feature_cols):
    """Cache predictions to avoid re-running model on every rerender."""
    input_df = pd.DataFrame([feature_values], columns=feature_cols)
    pred = int(_model.predict(input_df)[0])
    probs = _model.predict_proba(input_df)[0].tolist()
    return pred, probs


# ---- Load data ----
bundle, load_error = load_model_bundle()
raw_df = load_raw_dataset()

# ---- Header ----
st.title("🧠 Parkinson's Disease Speech Diagnostic AI")
st.caption("500-Tree Random Forest | 80/20 Subject-Grouped Split | CUDA-Trained, CPU-Served")

if load_error:
    st.error(f"**Startup Error:** {load_error}")
    st.info("Run `python train.py` to generate the model.")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
test_metrics = bundle["test_metrics"]
cv_results = bundle["cv_results"]
feature_importances = pd.DataFrame(bundle["feature_importances"])
feature_stats = bundle["feature_stats"]

# ---- Sidebar ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🔬 Patient Diagnosis",
    "📊 Feature Importance",
    "📈 Model Performance",
    "📁 Dataset Explorer"
])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Zero Data Leakage**\n\n"
    "Split by Patient Subject ID (80/20). "
    "Model evaluated on completely unseen patients."
)

# =============================================
# PAGE 1: PATIENT DIAGNOSIS
# =============================================
if page == "🔬 Patient Diagnosis":
    st.subheader("Patient Voice Biomarker Assessment")

    # Preset buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    if raw_df is not None:
        sample_hc = raw_df[raw_df["status"] == 0].iloc[0]
        sample_pd = raw_df[raw_df["status"] == 1].iloc[0]

    with col1:
        if st.button("🟢 Healthy Sample"):
            st.session_state["inputs"] = {c: float(sample_hc[c]) for c in feature_cols}
    with col2:
        if st.button("🔴 Parkinson's Sample"):
            st.session_state["inputs"] = {c: float(sample_pd[c]) for c in feature_cols}
    with col3:
        if st.button("🔄 Reset to Medians"):
            st.session_state["inputs"] = {c: float(feature_stats["all_median"][c]) for c in feature_cols}

    if "inputs" not in st.session_state:
        st.session_state["inputs"] = {c: float(feature_stats["all_median"][c]) for c in feature_cols}

    user_values = {}
    st.markdown("---")

    # Feature input tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Frequency (F0)", "2. Jitter", "3. Shimmer", "4. Nonlinear"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        user_values["MDVP:Fo(Hz)"] = c1.number_input("Avg F0 (Hz)", 50.0, 300.0, float(st.session_state["inputs"]["MDVP:Fo(Hz)"]), step=1.0)
        user_values["MDVP:Fhi(Hz)"] = c2.number_input("Max F0 (Hz)", 50.0, 600.0, float(st.session_state["inputs"]["MDVP:Fhi(Hz)"]), step=1.0)
        user_values["MDVP:Flo(Hz)"] = c3.number_input("Min F0 (Hz)", 50.0, 300.0, float(st.session_state["inputs"]["MDVP:Flo(Hz)"]), step=1.0)

    with tab2:
        c1, c2, c3 = st.columns(3)
        user_values["MDVP:Jitter(%)"] = c1.number_input("Jitter (%)", 0.0, 0.1, float(st.session_state["inputs"]["MDVP:Jitter(%)"]), format="%.5f")
        user_values["MDVP:RAP"] = c1.number_input("RAP", 0.0, 0.06, float(st.session_state["inputs"]["MDVP:RAP"]), format="%.5f")
        user_values["MDVP:Jitter(Abs)"] = c2.number_input("Jitter(Abs)", 0.0, 0.001, float(st.session_state["inputs"]["MDVP:Jitter(Abs)"]), format="%.6f")
        user_values["MDVP:PPQ"] = c2.number_input("PPQ", 0.0, 0.06, float(st.session_state["inputs"]["MDVP:PPQ"]), format="%.5f")
        user_values["Jitter:DDP"] = c3.number_input("Jitter:DDP", 0.0, 0.15, float(st.session_state["inputs"]["Jitter:DDP"]), format="%.5f")

    with tab3:
        c1, c2, c3 = st.columns(3)
        user_values["MDVP:Shimmer"] = c1.number_input("Shimmer", 0.0, 0.2, float(st.session_state["inputs"]["MDVP:Shimmer"]), format="%.5f")
        user_values["Shimmer:APQ5"] = c1.number_input("APQ5", 0.0, 0.15, float(st.session_state["inputs"]["Shimmer:APQ5"]), format="%.5f")
        user_values["MDVP:Shimmer(dB)"] = c2.number_input("Shimmer (dB)", 0.0, 1.5, float(st.session_state["inputs"]["MDVP:Shimmer(dB)"]), format="%.4f")
        user_values["MDVP:APQ"] = c2.number_input("APQ", 0.0, 0.15, float(st.session_state["inputs"]["MDVP:APQ"]), format="%.5f")
        user_values["Shimmer:APQ3"] = c3.number_input("APQ3", 0.0, 0.1, float(st.session_state["inputs"]["Shimmer:APQ3"]), format="%.5f")
        user_values["Shimmer:DDA"] = c3.number_input("DDA", 0.0, 0.3, float(st.session_state["inputs"]["Shimmer:DDA"]), format="%.5f")

    with tab4:
        c1, c2, c3 = st.columns(3)
        user_values["NHR"] = c1.number_input("NHR", 0.0, 0.5, float(st.session_state["inputs"]["NHR"]), format="%.5f")
        user_values["HNR"] = c1.number_input("HNR", 0.0, 40.0, float(st.session_state["inputs"]["HNR"]), format="%.2f")
        user_values["RPDE"] = c1.number_input("RPDE", 0.0, 1.0, float(st.session_state["inputs"]["RPDE"]), format="%.4f")
        user_values["DFA"] = c2.number_input("DFA", 0.4, 1.0, float(st.session_state["inputs"]["DFA"]), format="%.4f")
        user_values["spread1"] = c2.number_input("spread1", -10.0, 0.0, float(st.session_state["inputs"]["spread1"]), format="%.4f")
        user_values["spread2"] = c2.number_input("spread2", 0.0, 1.0, float(st.session_state["inputs"]["spread2"]), format="%.4f")
        user_values["D2"] = c3.number_input("D2", 1.0, 4.0, float(st.session_state["inputs"]["D2"]), format="%.4f")
        user_values["PPE"] = c3.number_input("PPE", 0.0, 0.6, float(st.session_state["inputs"]["PPE"]), format="%.4f")

    st.markdown("---")

    # Prediction with error handling
    try:
        feature_values = [user_values[c] for c in feature_cols]
        prediction, probabilities = run_prediction(model, feature_values, feature_cols)
        pd_prob = probabilities[1]
        hc_prob = probabilities[0]

        st.subheader("Diagnostic Prediction Result")
        r1, r2 = st.columns(2)

        with r1:
            if prediction == 1:
                st.error(f"**⚠️ Parkinson's Disease Detected (Class 1)**\n\n"
                         f"Probability of PD: **{pd_prob*100:.1f}%**\n\n"
                         f"Confidence: {(pd_prob - 0.5)*200:.1f}% above threshold")
            else:
                st.success(f"**✅ Healthy Vocal Profile (Class 0)**\n\n"
                           f"Probability Healthy: **{hc_prob*100:.1f}%**\n\n"
                           f"Probability PD: {pd_prob*100:.1f}%")

        with r2:
            st.write("**Probability Breakdown:**")
            st.bar_chart(
                pd.DataFrame({"Probability": [hc_prob, pd_prob]},
                             index=["Healthy (0)", "PD (1)"]),
                color="#3B82F6"
            )

        # Comparison table
        st.write("#### Key Biomarkers vs Cohort Medians")
        comp_rows = []
        for f in ["PPE", "spread1", "MDVP:Fo(Hz)", "HNR", "MDVP:Shimmer"]:
            comp_rows.append({
                "Biomarker": f,
                "Patient Value": round(user_values[f], 4),
                "Healthy Median": round(feature_stats["hc_median"][f], 4),
                "PD Median": round(feature_stats["pd_median"][f], 4)
            })
        st.table(pd.DataFrame(comp_rows))

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =============================================
# PAGE 2: FEATURE IMPORTANCE
# =============================================
elif page == "📊 Feature Importance":
    st.subheader("Acoustic Biomarker Importance")

    top_n = st.slider("Top features to show", 5, 22, 10)
    top_fi = feature_importances.head(top_n)

    st.bar_chart(top_fi.set_index("feature")["perm_importance_mean"], color="#10B981")

    st.write("#### Detailed Importance Table")
    display_cols = [c for c in top_fi.columns if c in ["feature", "perm_importance_mean", "perm_importance_std", "gain_importance", "gini_importance"]]
    st.table(top_fi[display_cols])

# =============================================
# PAGE 3: MODEL PERFORMANCE
# =============================================
elif page == "📈 Model Performance":
    st.subheader("Model Validation (Zero Data Leakage)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{test_metrics['recall']*100:.1f}%")
    c2.metric("Specificity", f"{test_metrics['specificity']*100:.1f}%")
    c3.metric("Balanced Acc", f"{test_metrics['balanced_accuracy']*100:.1f}%")
    c4.metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.write("#### Confusion Matrix")
        cm = np.array(test_metrics["confusion_matrix"])
        st.table(pd.DataFrame(
            cm,
            columns=["Pred Healthy", "Pred PD"],
            index=["True Healthy", "True PD"]
        ))
        st.caption(f"Test: {test_metrics['test_samples']} recordings, {test_metrics['test_subjects']} subjects")

    with right:
        st.write("#### 5-Fold Cross-Validation")
        cv_df = pd.DataFrame(cv_results)[["fold", "recall", "specificity", "balanced_accuracy", "roc_auc", "f1"]]
        means = cv_df.drop(columns="fold").mean()
        stds = cv_df.drop(columns="fold").std()
        summary = {col: f"{means[col]:.2f} ± {stds[col]:.2f}" for col in means.index}
        summary["fold"] = "Mean±Std"
        cv_display = pd.concat([cv_df, pd.DataFrame([summary])], ignore_index=True)
        st.table(cv_display)

# =============================================
# PAGE 4: DATASET EXPLORER
# =============================================
elif page == "📁 Dataset Explorer":
    st.subheader("Oxford Parkinson's Voice Dataset")

    if raw_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Recordings", len(raw_df))
        c2.metric("Subjects", raw_df["name"].apply(lambda x: "_".join(x.split("_")[:-1])).nunique())
        c3.metric("PD / Healthy", f"{(raw_df['status']==1).sum()} / {(raw_df['status']==0).sum()}")

        filt = st.selectbox("Filter", ["All", "Parkinson's (1)", "Healthy (0)"])
        if filt == "Parkinson's (1)":
            df_show = raw_df[raw_df["status"] == 1]
        elif filt == "Healthy (0)":
            df_show = raw_df[raw_df["status"] == 0]
        else:
            df_show = raw_df

        st.table(df_show.head(50))
        st.caption(f"Showing first {min(50, len(df_show))} of {len(df_show)} rows")
        st.download_button("Download CSV", df_show.to_csv(index=False), "parkinsons_filtered.csv")
    else:
        st.warning("parkinsons.csv not found.")
