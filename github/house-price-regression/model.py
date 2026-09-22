"""
House Price Prediction - Multiple Linear Regression
=====================================================
Features : Square Footage, Bedrooms, Bathrooms
Target   : Price (USD)

Model Equation (fitted on full dataset):
  Price = -31127.47
        + 169.24 * sqft
        + 6423.53 * bedrooms
        + 9005.35 * bathrooms

Full-dataset Metrics:
  R2               = 0.9898
  Adjusted R2      = 0.9892
  RMSE             = $12,355
  MAE              = $8,084
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")          # headless backend — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("house_prices.csv")
print("Dataset shape:", df.shape)
print("\nSummary statistics:")
print(df.describe().to_string())

# ── Features & target ─────────────────────────────────────────────────────────
FEATURES = ["sqft", "bedrooms", "bathrooms"]
TARGET   = "price"

X = df[FEATURES].values
y = df[TARGET].values

# ── Train / test split (80 / 20) ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── Fit model ─────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
for name, coef in zip(FEATURES, model.coef_):
    print(f"  {name:12s}: {coef:>12.4f}")
print(f"  {'intercept':12s}: {model.intercept_:>12.2f}")

# ── Evaluate on test set ───────────────────────────────────────────────────────
y_pred_test = model.predict(X_test)

r2   = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae  = mean_absolute_error(y_test, y_pred_test)

print(f"\nTest-set Metrics:")
print(f"  R2   = {r2:.4f}")
print(f"  RMSE = ${rmse:,.2f}")
print(f"  MAE  = ${mae:,.2f}")

# ── Predict a sample house ────────────────────────────────────────────────────
new_house = pd.DataFrame([[2000, 3, 2]], columns=FEATURES)
predicted_price = model.predict(new_house)[0]
print(f"\nSample prediction — 2000 sqft / 3 bed / 2 bath:")
print(f"  Predicted price: ${predicted_price:,.2f}")

# ── Visualisations ────────────────────────────────────────────────────────────
all_pred = model.predict(X)
residuals = y - all_pred

fig = plt.figure(figsize=(14, 10))
fig.suptitle("House Price — Linear Regression Diagnostics",
             fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Actual vs Predicted
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y, all_pred, alpha=0.7, edgecolors="white",
            linewidth=0.5, color="#4f8ef7", s=70, label="Houses")
lo, hi = min(y.min(), all_pred.min()), max(y.max(), all_pred.max())
ax1.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
ax1.set_xlabel("Actual Price ($)")
ax1.set_ylabel("Predicted Price ($)")
ax1.set_title(f"Actual vs Predicted  (R2 = {r2_score(y, all_pred):.4f})")
ax1.legend()
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# 2. Residuals
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(all_pred, residuals, alpha=0.7, color="#f97c4f",
            edgecolors="white", linewidth=0.5, s=70)
ax2.axhline(0, color="navy", linewidth=1.5, linestyle="--")
ax2.set_xlabel("Predicted Price ($)")
ax2.set_ylabel("Residual ($)")
ax2.set_title("Residual Plot")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# 3. Price vs Sqft
ax3 = fig.add_subplot(gs[1, 0])
sc = ax3.scatter(df["sqft"], df["price"], c=df["bedrooms"],
                 cmap="viridis", alpha=0.8, edgecolors="white",
                 linewidth=0.5, s=70)
plt.colorbar(sc, ax=ax3, label="Bedrooms")
ax3.set_xlabel("Square Footage")
ax3.set_ylabel("Price ($)")
ax3.set_title("Price vs Square Footage")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# 4. Price vs Bedrooms
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(df["bedrooms"], df["price"], alpha=0.7, color="#4ecb71",
            edgecolors="white", linewidth=0.5, s=70)
ax4.set_xlabel("Bedrooms")
ax4.set_ylabel("Price ($)")
ax4.set_title("Price vs Bedrooms")
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# 5. Price vs Bathrooms
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(df["bathrooms"], df["price"], alpha=0.7, color="#f7c948",
            edgecolors="white", linewidth=0.5, s=70)
ax5.set_xlabel("Bathrooms")
ax5.set_ylabel("Price ($)")
ax5.set_title("Price vs Bathrooms")
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

plt.savefig("plots.png", dpi=150, bbox_inches="tight")
print("\nDiagnostic plots saved as  plots.png")
