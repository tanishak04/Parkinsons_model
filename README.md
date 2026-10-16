# Parkinson's Disease Voice Detection - Random Forest Classifier

This project implements an end-to-end Machine Learning pipeline and an interactive **Streamlit web application** for detecting Parkinson's Disease (PD) from biomedical voice recordings (`parkinsons.csv`).

---

## 🌟 Key Features

1. **Zero Data Leakage Validation**:
   - The dataset contains 195 recordings from 32 distinct subjects (~6 recordings per subject).
   - Rather than naive random splitting, we use **Subject-Grouped Splitting (80% Train / 20% Test)** via `StratifiedGroupKFold`.
   - Voice recordings from the same individual never overlap between training and testing folds, guaranteeing true generalization to unseen patients.

2. **Cost-Sensitive Class Balancing**:
   - Accounts for the ~3:1 class imbalance (147 PD vs 48 Healthy) using balanced sample weighting.
   - Optimized for **Recall / Sensitivity (100% on unseen test set)** to minimize missed diagnoses (False Negatives).

3. **Interactive Streamlit Interface (`app.py`)**:
   - **🔬 Patient Diagnosis & Predictor**: Real-time diagnostic risk assessment with preset Healthy/PD samples and interactive sliders for 22 acoustic features.
   - **📊 Biomarker Feature Importance**: Permutation and Gini importance rankings highlighting top dysphonia indicators (`PPE`, `spread1`, `MDVP:Fo(Hz)`).
   - **📈 Model Performance Hub**: Confusion matrix, ROC-AUC, and 5-Fold Stratified Group CV results table.
   - **📁 Dataset Explorer**: Search, filter, and inspect the raw Oxford dataset.

---

## 🚀 Quickstart

### 1. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
This opens the interactive diagnostic dashboard in your web browser (default: `http://localhost:8501`).

### 2. Run Single Sample CLI Predictions
```powershell
python predict.py
```
Or pass a JSON file with feature values:
```powershell
python predict.py patient_data.json
```

### 3. Retrain the Model on CUDA GPU
```powershell
python train.py
```
This automatically detects your **NVIDIA GeForce RTX 5050 GPU** and accelerates parallel tree construction using CUDA cores via `device="cuda"`.

---

## 📊 Performance Summary (Subject-Grouped 80/20 Holdout)

- **Compute Hardware**: **NVIDIA GeForce RTX 5050 (CUDA)**
- **Model**: 500-Tree Parallel Random Forest (`num_parallel_tree=500`, `device='cuda'`)
- **Total Samples**: 195 (32 unique subjects)
- **Train Set (80%)**: 158 recordings across 26 subjects
- **Unseen Test Set (20%)**: 37 recordings across 6 held-out subjects
- **Test Accuracy**: **89.2%**
- **Test Recall (Sensitivity)**: **100%** (31 / 31 Parkinson's cases detected with 0 false negatives)
- **Test F1-Score**: **0.939**
- **Test Balanced Accuracy**: **66.7%**
- **Test Specificity**: **33.3%**


---

## 📁 Repository Structure

```
parkinsons/
├── app.py                     # Interactive Streamlit Web Application
├── train.py                   # Data loading, grouped CV, training & evaluation pipeline
├── predict.py                 # CLI inference script
├── parkinsons.csv             # Oxford Parkinson's Voice Dataset
├── models/
│   └── rf_model.joblib        # Trained Random Forest model & metadata bundle
├── reports/
│   └── evaluation_report.json # Detailed test & cross-validation metrics
└── README.md                  # Project documentation
```
