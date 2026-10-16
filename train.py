import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

def check_cuda_availability():
    try:
        dummy_X = np.ones((5, 2), dtype=np.float32)
        dummy_y = np.array([0, 1, 0, 1, 0])
        test_clf = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda")
        test_clf.fit(dummy_X, dummy_y)
        return "cuda"
    except Exception as e:
        print(f"CUDA check failed ({e}), falling back to CPU.")
        return "cpu"

def load_and_prepare_data(csv_path="parkinsons.csv"):
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["name"].apply(lambda x: "_".join(x.split("_")[:-1]))
    feature_cols = [c for c in df.columns if c not in ["name", "status", "subject_id"]]
    X = df[feature_cols]
    y = df["status"]
    groups = df["subject_id"]
    return df, X, y, groups, feature_cols

def build_gpu_random_forest(device="cuda", random_state=42, n_trees=300):
    # Construct a true Breiman-style Random Forest using GPU CUDA acceleration in XGBoost
    return xgb.XGBClassifier(
        n_estimators=1,
        num_parallel_tree=n_trees,
        subsample=0.8,
        colsample_bynode=0.8,
        learning_rate=1.0,
        max_depth=6,
        tree_method="hist",
        device=device,
        scale_pos_weight=0.33, # Adjust for 75% PD vs 25% Healthy class imbalance
        random_state=random_state,
        eval_metric="logloss"
    )

def run_cross_validation(X, y, groups, device="cuda", n_splits=5, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = build_gpu_random_forest(device=device, random_state=random_state + fold)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        fold_metrics.append({
            "fold": fold,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) > 1 else np.nan,
            "train_subjects": int(groups.iloc[train_idx].nunique()),
            "val_subjects": int(groups.iloc[val_idx].nunique()),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx)
        })
    return fold_metrics

def train_and_evaluate_holdout(X, y, groups, feature_cols, device="cuda", random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    test_groups = groups.iloc[test_idx]
    
    model = build_gpu_random_forest(device=device, random_state=random_state, n_trees=500)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    test_metrics = {
        "device": device,
        "gpu_accelerated": (device == "cuda"),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_subjects": int(groups.iloc[train_idx].nunique()),
        "test_subjects": int(test_groups.nunique()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist()
    }
    
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state)
    feature_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain_importance": model.feature_importances_,
        "perm_importance_mean": perm_imp.importances_mean,
        "perm_importance_std": perm_imp.importances_std
    }).sort_values("perm_importance_mean", ascending=False)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    
    curve_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist()
    }
    
    return model, test_metrics, feature_imp_df, curve_data, X_test, y_test, test_groups

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    device = check_cuda_availability()
    print(f"\n==================================================")
    print(f"Hardware Compute Device: {device.upper()} (GPU Acceleration: {device == 'cuda'})")
    print(f"==================================================")
    
    print("\nLoading dataset parkinsons.csv...")
    df, X, y, groups, feature_cols = load_and_prepare_data()
    print(f"Total samples: {len(df)}, Unique subjects: {groups.nunique()}")
    
    print(f"\nRunning 5-Fold Stratified Group Cross-Validation on {device.upper()}...")
    cv_results = run_cross_validation(X, y, groups, device=device, n_splits=5)
    cv_df = pd.DataFrame(cv_results)
    
    print("\n--- 5-Fold Stratified Group CV Summary ---")
    print(f"Mean Recall:            {cv_df['recall'].mean():.3f} +/- {cv_df['recall'].std():.3f}")
    print(f"Mean Specificity:       {cv_df['specificity'].mean():.3f} +/- {cv_df['specificity'].std():.3f}")
    print(f"Mean Balanced Accuracy: {cv_df['balanced_accuracy'].mean():.3f} +/- {cv_df['balanced_accuracy'].std():.3f}")
    print(f"Mean ROC-AUC:           {cv_df['roc_auc'].mean():.3f} +/- {cv_df['roc_auc'].std():.3f}")
    
    print(f"\nTraining 500-Tree Random Forest on {device.upper()} (80% Train, evaluating on 20% Unseen Patients)...")
    model, test_metrics, feat_imp, curve_data, X_test, y_test, test_groups = train_and_evaluate_holdout(
        X, y, groups, feature_cols, device=device
    )
    
    print("\n--- Unseen Subject Test Set Performance ---")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    print("Confusion Matrix (TN, FP / FN, TP):")
    print(np.array(test_metrics["confusion_matrix"]))
    
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
        "cv_results": cv_results,
        "feature_importances": feat_imp.to_dict(orient="records"),
        "curve_data": curve_data,
        "device": device,
        "feature_stats": {
            "all_mean": X.mean().to_dict(),
            "all_median": X.median().to_dict(),
            "all_min": X.min().to_dict(),
            "all_max": X.max().to_dict(),
            "pd_median": X[y == 1].median().to_dict(),
            "hc_median": X[y == 0].median().to_dict()
        }
    }
    
    model_path = os.path.join("models", "rf_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"\nModel and evaluation bundle saved to: {model_path}")
    
    metrics_path = os.path.join("reports", "evaluation_report.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "device": device,
            "test_metrics": test_metrics,
            "cv_results": cv_results,
            "top_features": feat_imp.head(10).to_dict(orient="records")
        }, f, indent=2)
    print(f"Evaluation report saved to: {metrics_path}")

if __name__ == "__main__":
    main()
