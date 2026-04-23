"""
SHAP Explainability Report for UVCRS Random Forest Recommender
Generates visualizations saved to reports/shap/
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

OUT_DIR = "reports/shap"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORICAL_FEATURES = ["make", "bodytype", "Fuel Type", "Drivetrain", "Transmission Type"]

# ── Load artifacts ──────────────────────────────────────────────────────────
print("Loading model and data...")
model = joblib.load("models/rf_recommender.joblib")
label_encoders = joblib.load("models/label_encoders.joblib")
with open("models/feature_list.json") as f:
    FEATURES = json.load(f)

# Try both file extensions used in the project
data_path = "data/final/model_ready.csv"
if not os.path.exists(data_path):
    data_path = "data/final/model_ready.UVCRS"
df = pd.read_csv(data_path)

# Apply label encoding (same as train_recommender.py)
for col in CATEGORICAL_FEATURES:
    le = label_encoders[col]
    known = set(le.classes_)
    df[col + "_enc"] = df[col].astype(str).apply(
        lambda v, le=le, known=known: le.transform([v])[0] if v in known else -1
    )

X = df[FEATURES].copy()
y = df["consumer_overall_rating"]

# Sample for SHAP (full dataset is slow; 2000 rows is representative)
rng = np.random.default_rng(42)
idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
X_sample = X.iloc[idx].reset_index(drop=True)
y_sample = y.iloc[idx].reset_index(drop=True)

print(f"Running SHAP TreeExplainer on {len(X_sample)} samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# expected_value can be a 0-d array or 1-element array depending on shap version
base_value = float(np.asarray(explainer.expected_value).flat[0])

# ── 1. Global bar chart (mean |SHAP|) ────────────────────────────────────────
print("Saving 1/6 — global_bar.png")
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
plt.title("Mean |SHAP Value| — Top 20 Features", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/global_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Beeswarm summary plot ─────────────────────────────────────────────────
print("Saving 2/6 — beeswarm_summary.png")
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.title("SHAP Beeswarm — Feature Impact on Predicted Rating", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/beeswarm_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Dependence plots for top 6 features ───────────────────────────────────
mean_abs = np.abs(shap_values).mean(axis=0)
top6_idx = np.argsort(mean_abs)[::-1][:6]
top6_features = [FEATURES[i] for i in top6_idx]

print("Saving 3/6 — dependence_top6.png")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feat in zip(axes.flat, top6_features):
    plt.sca(ax)
    shap.dependence_plot(feat, shap_values, X_sample, ax=ax, show=False)
    ax.set_title(f"Dependence: {feat}", fontsize=10)
plt.suptitle("SHAP Dependence Plots — Top 6 Features", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dependence_top6.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Waterfall — highest-rated car in sample ───────────────────────────────
print("Saving 4/6 — waterfall_best_car.png")
best_i = int(y_sample.idxmax())
exp = shap.Explanation(
    values=shap_values[best_i],
    base_values=base_value,
    data=X_sample.iloc[best_i].values,
    feature_names=FEATURES,
)
shap.waterfall_plot(exp, max_display=15, show=False)
car_label = ""
if "make" in df.columns and "model" in df.columns:
    orig_i = idx[best_i]
    car_label = f" — {df.iloc[orig_i]['make']} {df.iloc[orig_i]['model']} ({int(df.iloc[orig_i]['year'])})"
plt.title(f"Waterfall: Highest-Rated Car{car_label}", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/waterfall_best_car.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Waterfall — lowest-rated car in sample ────────────────────────────────
print("Saving 5/6 — waterfall_worst_car.png")
worst_i = int(y_sample.idxmin())
exp_w = shap.Explanation(
    values=shap_values[worst_i],
    base_values=base_value,
    data=X_sample.iloc[worst_i].values,
    feature_names=FEATURES,
)
shap.waterfall_plot(exp_w, max_display=15, show=False)
car_label_w = ""
if "make" in df.columns and "model" in df.columns:
    orig_w = idx[worst_i]
    car_label_w = f" — {df.iloc[orig_w]['make']} {df.iloc[orig_w]['model']} ({int(df.iloc[orig_w]['year'])})"
plt.title(f"Waterfall: Lowest-Rated Car{car_label_w}", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/waterfall_worst_car.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6. Heatmap — SHAP value matrix (top 15 features, 200 cars) ───────────────
print("Saving 6/6 — heatmap.png")
top15_idx = np.argsort(mean_abs)[::-1][:15]
top15_features = [FEATURES[i] for i in top15_idx]
shap_top15 = shap_values[:200, top15_idx]

fig, ax = plt.subplots(figsize=(16, 10))
vmax = np.percentile(np.abs(shap_top15), 98)
im = ax.imshow(shap_top15.T, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_yticks(range(len(top15_features)))
ax.set_yticklabels(top15_features, fontsize=9)
ax.set_xlabel("Car (sample index, first 200)", fontsize=11)
ax.set_title("SHAP Value Heatmap — Top 15 Features x 200 Cars", fontsize=13, pad=10)
plt.colorbar(im, ax=ax, label="SHAP Value")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Summary CSV ───────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs,
    "mean_shap": shap_values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
summary.to_csv(f"{OUT_DIR}/shap_feature_importance.csv", index=False)

print(f"\nAll outputs saved to {OUT_DIR}/")
print("\nTop 10 most impactful features:")
print(summary.head(10).to_string(index=False))
