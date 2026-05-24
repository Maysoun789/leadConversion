import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset.csv"

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================

print("=" * 70)
print("STEP 1: DATA LOADING & EXPLORATION")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['converted'].value_counts()}")
print(f"\nTarget distribution (%):\n{df['converted'].value_counts(normalize=True) * 100}")

# ============================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# Boolean features
df["Demande_Devis_enc"] = df["Demande Devis"].map({"TRUE": 1, "FALSE": 0})
df["Connait_Marque_enc"] = df["Connait Marque"].map({"TRUE": 1, "FALSE": 0})

# Ordinal encoding for Interest Level
interest_map = {
    "Froid": 0,
    "Tiède": 1,
    "Chaud": 2,
}

df["Interest_Level_enc"] = df["Interest Level"].map(interest_map)

# One-hot encoding for Decision Level
decision_dummies = pd.get_dummies(df["Decision Level"], prefix="Decision", drop_first=False)

# TF-IDF for Meeting Notes
tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

tfidf_matrix = tfidf.fit_transform(df["Meeting Notes"].fillna(""))
tfidf_feature_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]

print(f"TF-IDF features extracted: {tfidf_matrix.shape[1]}")

# Text-derived features
df["notes_length"] = df["Meeting Notes"].fillna("").apply(len)
df["notes_word_count"] = df["Meeting Notes"].fillna("").apply(lambda x: len(x.split()))

positive_keywords = [
    "budget",
    "urgent",
    "signé",
    "motivé",
    "commande",
    "qualité",
    "rapide",
    "chantier",
    "disponible",
]

negative_keywords = [
    "trop cher",
    "concurrent",
    "hésit",
    "pas de",
    "difficile",
    "rupture",
    "limité",
]

for kw in positive_keywords:
    df[f"kw_{kw}"] = (
        df["Meeting Notes"]
        .fillna("")
        .str.lower()
        .str.contains(kw, na=False)
        .astype(int)
    )

for kw in negative_keywords:
    df[f"kw_neg_{kw.replace(' ', '_')}"] = (
        df["Meeting Notes"]
        .fillna("")
        .str.lower()
        .str.contains(kw, na=False)
        .astype(int)
    )

df["positive_kw_count"] = df[[f"kw_{kw}" for kw in positive_keywords]].sum(axis=1)
df["negative_kw_count"] = df[
    [f"kw_neg_{kw.replace(' ', '_')}" for kw in negative_keywords]
].sum(axis=1)

# Combine features
numeric_features = [
    "Demande_Devis_enc",
    "Connait_Marque_enc",
    "Interest_Level_enc",
    "notes_length",
    "notes_word_count",
    "positive_kw_count",
    "negative_kw_count",
]

keyword_features = [c for c in df.columns if c.startswith("kw_")]

X_numeric = df[numeric_features + keyword_features].values.astype(np.float64)
X_decision = decision_dummies.values.astype(np.float64)
X_tfidf = tfidf_matrix.toarray().astype(np.float64)

X_combined = np.hstack([X_numeric, X_decision, X_tfidf])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

y = df["converted"].values

feature_names = (
    numeric_features
    + keyword_features
    + list(decision_dummies.columns)
    + tfidf_feature_names
)

print(f"\nFinal feature matrix shape: {X_combined.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: MODEL TRAINING & EVALUATION")
print("=" * 70)

models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
results = {}
trained_models = {}
model_names = list(models.keys())

for name, model in models.items():
    print(f"\n--- {name} ---")

    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    cv_results[name] = {
        "cv_accuracy_mean": cv_acc.mean(),
        "cv_accuracy_std": cv_acc.std(),
        "cv_f1_mean": cv_f1.mean(),
        "cv_f1_std": cv_f1.std(),
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std(),
    }

    print(f"  CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"  CV F1 Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV AUC-ROC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    trained_models[name] = model

    print(f"  Test Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  Test F1 Score:  {results[name]['f1_score']:.4f}")
    print(f"  Test AUC-ROC:   {results[name]['auc_roc']:.4f}")
    print(f"  Confusion Matrix:\n{results[name]['confusion_matrix']}")

# Best model
best_model_name = max(results.keys(), key=lambda k: results[k]["f1_score"])

print(
    f"\n*** BEST MODEL: {best_model_name} "
    f"(F1={results[best_model_name]['f1_score']:.4f}) ***"
)

# ============================================================
# SAVE MODEL BUNDLE FOR RAILWAY API
# ============================================================

model_bundle = {
    "model": trained_models[best_model_name],
    "best_model_name": best_model_name,
    "scaler": scaler,
    "tfidf": tfidf,
    "decision_columns": list(decision_dummies.columns),
    "feature_names": feature_names,
    "numeric_features": numeric_features,
    "keyword_features": keyword_features,
    "positive_keywords": positive_keywords,
    "negative_keywords": negative_keywords,
    "interest_map": interest_map,
}

joblib.dump(model_bundle, "model.pkl")

print("\nModel bundle saved as model.pkl")

# ============================================================
# 4. COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: RESULTS COMPARISON TABLE")
print("=" * 70)

comparison_data = []

for name in model_names:
    r = results[name]
    cv_item = cv_results[name]

    comparison_data.append(
        {
            "Model": name,
            "CV Acc": f"{cv_item['cv_accuracy_mean']:.4f}+/-{cv_item['cv_accuracy_std']:.4f}",
            "CV F1": f"{cv_item['cv_f1_mean']:.4f}+/-{cv_item['cv_f1_std']:.4f}",
            "CV AUC": f"{cv_item['cv_auc_mean']:.4f}+/-{cv_item['cv_auc_std']:.4f}",
            "Test Acc": f"{r['accuracy']:.4f}",
            "Test Prec": f"{r['precision']:.4f}",
            "Test Recall": f"{r['recall']:.4f}",
            "Test F1": f"{r['f1_score']:.4f}",
            "Test AUC": f"{r['auc_roc']:.4f}",
        }
    )

comparison_df = pd.DataFrame(comparison_data)

print(comparison_df.to_string(index=False))

# ============================================================
# 5. VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: GENERATING VISUALIZATIONS")
print("=" * 70)

colors = {
    "XGBoost": "#1f77b4",
    "LightGBM": "#ff7f0e",
    "Random Forest": "#2ca02c",
}

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 8))

for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    ax.plot(
        fpr,
        tpr,
        color=colors[name],
        lw=2.5,
        label=f'{name} (AUC = {results[name]["auc_roc"]:.4f})',
    )

ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves - Lead Conversion Classification", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "roc_curves_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - ROC curves saved")

# --- Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(10, 8))

for name in model_names:
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, results[name]["y_proba"])
    ax.plot(
        rec_vals,
        prec_vals,
        color=colors[name],
        lw=2.5,
        label=f'{name} (AP = {results[name]["avg_precision"]:.4f})',
    )

ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_title(
    "Precision-Recall Curves - Lead Conversion Classification",
    fontsize=15,
    fontweight="bold",
)
ax.legend(loc="best", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "precision_recall_curves.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - Precision-Recall curves saved")

# --- Metrics Comparison Bar Chart ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_keys = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc_roc",
    "avg_precision",
]

metrics_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC-ROC",
    "Avg Precision",
]

x = np.arange(len(model_names))

for idx, (metric, label) in enumerate(zip(metrics_keys, metrics_labels)):
    row, col = idx // 3, idx % 3

    values = [results[name][metric] for name in model_names]

    bars = axes[row, col].bar(
        x,
        values,
        0.5,
        color=[colors[n] for n in model_names],
        edgecolor="black",
        linewidth=0.5,
    )

    axes[row, col].set_ylabel(label, fontsize=11)
    axes[row, col].set_title(label, fontsize=13, fontweight="bold")
    axes[row, col].set_xticks(x)
    axes[row, col].set_xticklabels(model_names, fontsize=10)
    axes[row, col].set_ylim(0, 1.05)
    axes[row, col].grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        axes[row, col].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

fig.suptitle(
    "Model Performance Comparison - Lead Conversion",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)

plt.tight_layout()
plt.savefig(output_dir / "metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - Metrics comparison saved")

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, name in enumerate(model_names):
    cm = results[name]["confusion_matrix"]

    axes[idx].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[idx].set_title(f"{name}", fontsize=14, fontweight="bold")
    axes[idx].set_xlabel("Predicted", fontsize=11)
    axes[idx].set_ylabel("Actual", fontsize=11)
    axes[idx].set_xticks([0, 1])
    axes[idx].set_yticks([0, 1])
    axes[idx].set_xticklabels(["Not Converted", "Converted"], fontsize=9)
    axes[idx].set_yticklabels(["Not Converted", "Converted"], fontsize=9)

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[idx].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

fig.suptitle("Confusion Matrices Comparison", fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - Confusion matrices saved")

# --- CV Stability ---
fig, ax = plt.subplots(figsize=(12, 6))

cv_metrics = [
    "cv_accuracy_mean",
    "cv_f1_mean",
    "cv_auc_mean",
]

cv_labels = [
    "CV Accuracy",
    "CV F1",
    "CV AUC-ROC",
]

x = np.arange(len(cv_labels))
width = 0.2

for i, name in enumerate(model_names):
    means = [cv_results[name][m] for m in cv_metrics]
    stds = [cv_results[name][m.replace("_mean", "_std")] for m in cv_metrics]

    bars = ax.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        label=name,
        color=colors[name],
        edgecolor="black",
        linewidth=0.5,
        capsize=5,
    )

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Cross-Validation Performance (5-Fold)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(cv_labels, fontsize=11)
ax.legend(loc="best", fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "cv_stability.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - CV stability chart saved")

# --- Feature Importance best model ---
fig, ax = plt.subplots(figsize=(12, 8))

best_model = trained_models[best_model_name]
importances = best_model.feature_importances_

n_top = 20
indices = np.argsort(importances)[::-1][:n_top]

top_features = [
    feature_names[i] if i < len(feature_names) else f"feature_{i}"
    for i in indices
]

top_importances = importances[indices]

ax.barh(
    range(n_top),
    top_importances[::-1],
    color=colors[best_model_name],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_yticks(range(n_top))
ax.set_yticklabels(top_features[::-1], fontsize=10)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title(f"Top 20 Features - {best_model_name}", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(output_dir / "feature_importance_best_model.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"  - Feature importance ({best_model_name}) saved")

# --- Feature Importance all models ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

for idx, name in enumerate(model_names):
    imps = trained_models[name].feature_importances_
    top_idx = np.argsort(imps)[::-1][:15]

    top_f = [
        feature_names[i] if i < len(feature_names) else f"feature_{i}"
        for i in top_idx
    ]

    top_v = imps[top_idx]

    axes[idx].barh(
        range(15),
        top_v[::-1],
        color=colors[name],
        edgecolor="black",
        linewidth=0.5,
    )

    axes[idx].set_yticks(range(15))
    axes[idx].set_yticklabels(top_f[::-1], fontsize=8)
    axes[idx].set_xlabel("Importance", fontsize=11)
    axes[idx].set_title(f"{name}", fontsize=14, fontweight="bold")
    axes[idx].grid(True, alpha=0.3, axis="x")

fig.suptitle("Feature Importance Comparison (Top 15)", fontsize=16, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig(output_dir / "feature_importance_all_models.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - Feature importance all models saved")

# --- Radar Chart ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

categories = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC-ROC",
    "Avg Precision",
]

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for name in model_names:
    values = [results[name][m] for m in metrics_keys]
    values += values[:1]

    ax.plot(
        angles,
        values,
        "o-",
        linewidth=2.5,
        label=name,
        color=colors[name],
    )

    ax.fill(angles, values, alpha=0.1, color=colors[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0.5, 0.8)
ax.set_title("Model Performance Radar Chart", fontsize=14, fontweight="bold", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "radar_chart.png", dpi=200, bbox_inches="tight")
plt.close()

print("  - Radar chart saved")

# ============================================================
# 6. SAVE RESULTS TO JSON
# ============================================================

report_data = {
    "dataset_info": {
        "total_samples": int(df.shape[0]),
        "n_features": int(X_combined.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "target_distribution": {
            "class_0": int(sum(y == 0)),
            "class_1": int(sum(y == 1)),
            "class_0_pct": round(float(sum(y == 0) / len(y) * 100), 2),
            "class_1_pct": round(float(sum(y == 1) / len(y) * 100), 2),
        },
    },
    "feature_breakdown": {
        "numeric_keyword_features": len(numeric_features) + len(keyword_features),
        "decision_level_onehot": int(X_decision.shape[1]),
        "tfidf_features": int(X_tfidf.shape[1]),
        "total_features": int(X_combined.shape[1]),
    },
    "cv_results": {},
    "test_results": {},
    "best_model": best_model_name,
    "best_model_metrics": {
        "f1_score": round(float(results[best_model_name]["f1_score"]), 4),
        "auc_roc": round(float(results[best_model_name]["auc_roc"]), 4),
        "accuracy": round(float(results[best_model_name]["accuracy"]), 4),
        "precision": round(float(results[best_model_name]["precision"]), 4),
        "recall": round(float(results[best_model_name]["recall"]), 4),
    },
}

for name in model_names:
    report_data["cv_results"][name] = {
        k: round(float(v), 4) for k, v in cv_results[name].items()
    }

    report_data["test_results"][name] = {
        "accuracy": round(float(results[name]["accuracy"]), 4),
        "precision": round(float(results[name]["precision"]), 4),
        "recall": round(float(results[name]["recall"]), 4),
        "f1_score": round(float(results[name]["f1_score"]), 4),
        "auc_roc": round(float(results[name]["auc_roc"]), 4),
        "avg_precision": round(float(results[name]["avg_precision"]), 4),
        "confusion_matrix": results[name]["confusion_matrix"].tolist(),
    }

with open(output_dir / "classification_results.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)

print(f"\nBest Model: {best_model_name}")
print(f"  F1 Score:  {results[best_model_name]['f1_score']:.4f}")
print(f"  AUC-ROC:   {results[best_model_name]['auc_roc']:.4f}")
print(f"  Accuracy:  {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision: {results[best_model_name]['precision']:.4f}")
print(f"  Recall:    {results[best_model_name]['recall']:.4f}")

print("\nGenerated files:")
print("  - model.pkl")
print("  - output/classification_results.json")
print("  - output/roc_curves_comparison.png")
print("  - output/precision_recall_curves.png")
print("  - output/metrics_comparison.png")
print("  - output/confusion_matrices.png")
print("  - output/cv_stability.png")
print("  - output/feature_importance_best_model.png")
print("  - output/feature_importance_all_models.png")
print("  - output/radar_chart.png")