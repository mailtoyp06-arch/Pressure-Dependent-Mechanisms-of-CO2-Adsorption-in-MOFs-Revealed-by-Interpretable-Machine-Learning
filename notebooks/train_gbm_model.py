# train_gbm_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# 1. Load merged dataset
# -------------------------------------------------------
#df = pd.read_csv("validation_MOFs_texturals_with_CO2_Loading.csv")
#print("Dataset loaded successfully.")
#print(f"Shape: {df.shape}")
#print(f"Columns: {df.columns.tolist()}\n")

# -------------------------------------------------------
# 1. Load pre-split dataset
# -------------------------------------------------------
train_df = pd.read_csv("train_set.csv")
val_df   = pd.read_csv("validation_set.csv")
test_df  = pd.read_csv("test_set.csv")

print("Datasets loaded successfully.")
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}\n")

# -------------------------------------------------------
# 2. Data Cleaning
# -------------------------------------------------------
train_df = train_df.dropna()
val_df   = val_df.dropna()
test_df  = test_df.dropna()


print(f"Train rows after cleaning: {len(train_df)}")
print(f"Val rows after cleaning:   {len(val_df)}")
print(f"Test rows after cleaning:  {len(test_df)}")


# -------------------------------------------------------
# 3. Define features (X) and target (y)
# -------------------------------------------------------
# Adjust column names based on your dataset
# Target: CO₂ loading (mol/kg framework)
target_col =  "CO2 Loading (mol/kg)"
print(train_df.columns.tolist())

def split_X_y(df, target_col="Loading"):
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Always create log_fugacity
    if "Fugacity [Pa]" in X.columns:
        X["log_fugacity"] = np.log1p(X["Fugacity [Pa]"])

    # Always drop raw fugacity
    X = X.drop(columns=["Fugacity [Pa]"], errors="ignore")

    return X, y

X_train, y_train = split_X_y(train_df)
X_val, y_val     = split_X_y(val_df)
X_test, y_test   = split_X_y(test_df)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# -------------------------------------------------------
# 4. Split data (80/20)
# -------------------------------------------------------
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# 5. Scale numeric features
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled =  scaler.transform(X_val)
# -------------------------------------------------------
# 6. Train default Gradient Boosting Regressor
# -------------------------------------------------------
default_model = GradientBoostingRegressor(
random_state=42,
    n_estimators=500,         # moderate boosting length
    learning_rate=0.05,       # controlled learning
    max_depth=2,              # shallow trees generalize better
    subsample=0.7,            # stochastic training (reduces overfitting)
    min_samples_leaf=5,       # prevents tiny tree partitions
)
default_model.fit(X_train_scaled, y_train)


train_loss = []
val_loss = []

for y_train_pred, y_val_pred in zip(
        default_model.staged_predict(X_train_scaled),
	default_model.staged_predict(X_val_scaled)
            ):
    train_loss.append(mean_squared_error(y_train, y_train_pred))
    val_loss.append(mean_squared_error(y_val, y_val_pred))


y_val_pred = default_model.predict(X_val_scaled)

mse_val = mean_squared_error(y_val, y_val_pred)
r2_val  = r2_score(y_val, y_val_pred)

print("\nDefault Model Validation Performance:")
print(f"Val MSE:  {mse_val:.4f}, RMSE: {np.sqrt(mse_val):.4f}, R²: {r2_val:.4f}")

# -------------------------------------------------------
# 7. Cross-validation (5-fold)
# -------------------------------------------------------
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#cv_r2 = cross_val_score(default_model, X_train_scaled, y_train, cv=kf, scoring='r2')
#cv_rmse = np.sqrt(-cross_val_score(default_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error'))

#print(f"\nCross-Validation Results (Default Model):")
#print(f"Mean R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
#print(f"Mean RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# -------------------------------------------------------
# 8. GridSearchCV for Hyperparameter Tuning
# -------------------------------------------------------
param_grid = {
    'n_estimators':[ 300, 500,800],
    'learning_rate': [.01, .05, .1],
    'max_depth': [2, 3, 4],
    'subsample': [.5, .7, 1.0],
    'min_samples_leaf': [ 3, 5, 10]
}
grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("\nRunning GridSearchCV (this may take a few minutes)...")
grid_search.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# -------------------------------------------------------
# 9. Evaluate Tuned Model
# -------------------------------------------------------
best_model = grid_search.best_estimator_

y_test_pred = best_model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test  = r2_score(y_test, y_test_pred)

print("\nFinal Test Performance (Tuned Model):")
print(f"Test MSE:  {mse_test:.4f}, RMSE: {np.sqrt(mse_test):.4f}, R²: {r2_test:.4f}")

joblib.dump(best_model, "gbm_model.pkl")
joblib.dump(scaler, "gbm_scaler.pkl")

print("Saved: gbm_model.pkl")
print("Saved: gbm_scaler.pkl")
# -------------------------------------------------------
# 10. Feature Importance
# -------------------------------------------------------
importances = best_model.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top20 = importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(8, 6))
plt.barh(top20['Feature'], top20['Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.title("Top 20 Feature Importances — GBM", fontsize=14)

plt.tight_layout()
plt.savefig("gbm_top20_feature_importance.png", dpi=300)
plt.show()

print("Saved: gbm_top20_feature_importance.png")
print("\nTop 20 GBM Features:")
print(top20)
y_pred_best = best_model.predict(X_test_scaled)
#--------------------------------------------------------
# 10. Scatter Plot 
#--------------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.7, color='blue')
plt.xlabel("True CO₂ Loading")
plt.ylabel("Predicted CO₂ Loading")
plt.title("Predicted vs True CO₂ Loading")

# Add diagonal line for perfect prediction
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

plt.legend()
plt.tight_layout()
plt.savefig("prediction_vs_true.png")  # saves plot as PNG
plt.show()
#-------------------------------------------------------
# 11. Train  VS test Error Comparison Plot 
#------------------------------------------------------
y_train_pred = best_model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)

plt.figure(figsize=(6,5))
plt.bar(['Train RMSE', 'Test RMSE'], 
        [np.sqrt(mse_train), np.sqrt(mse_test)], 
        color=['skyblue', 'orange'])


plt.ylabel("RMSE")
plt.title("Train vs Test Error (GBM)")
plt.tight_layout()
plt.savefig("gbm_train_vs_test_rmse.png", dpi=300)
plt.show()
#--------------------------------------------------------
# 12. Cross - Validation Error Bar Plot 
#---------------------------------------------------------
#plt.figure(figsize=(6,5))
#plt.errorbar(
#    x=[0], 
#    y=[cv_r2.mean()], 
#    yerr=[cv_r2.std()], 
#    fmt='o', capsize=5
#)
#plt.xticks([0], ['GBM'])
#plt.ylabel("R² Score")
#plt.title("Cross-Validation R² with Error Bars (GBM)")
#plt.tight_layout()
#plt.savefig("gbm_cv_r2_errorbar.png", dpi=300)
#plt.close()
#if want to use 12, uncomment section 7  
#-------------------------------------------------------
#13.  Redisual Plot ( True - Predicted vs Predicted ) 
# ------------------------------------------------------
gbm_train_pred = best_model.predict(X_train_scaled)
gbm_test_pred  = best_model.predict(X_test_scaled)
residuals_gbm_train= y_train - gbm_train_pred
residuals_gbm_test = y_test  - gbm_test_pred

plt.figure(figsize=(6,5))
plt.scatter(gbm_train_pred, residuals_gbm_train, alpha=0.6, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted CO₂ Loading")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot (GBM) – Train")
plt.tight_layout()
plt.savefig("gbm_residuals_train.png", dpi=300)
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(gbm_test_pred, residuals_gbm_test, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted CO₂ Loading")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot (GBM) – Test")
plt.tight_layout()
plt.savefig("gbm_residuals_test.png", dpi=300)
plt.show()

#-------------------------------------------------------
#14. Learning Curve PLot 
#-------------------------------------------------------
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_scaled, y_train,
    cv=5, scoring='r2',
    train_sizes=np.linspace(0.2, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(6,5))
plt.plot(train_sizes, train_mean, marker='o', label='Train R²')
plt.plot(train_sizes, test_mean, marker='s', label='CV R²')

plt.xlabel("Training Samples Used")
plt.ylabel("R² Score")
plt.title("Learning Curve (GBM)")
plt.legend()
plt.tight_layout()
plt.savefig("gbm_learning_curve.png", dpi=300)
plt.show()

#------------------------------------------------------
#15. Validation loss vs boosting iteration 
#------------------------------------------------------
from sklearn.metrics import mean_squared_error

plt.figure(figsize=(7,5))
plt.plot(train_loss, label="Training MSE")
plt.plot(val_loss, label="Validation MSE")
plt.xlabel("Boosting Iterations")
plt.ylabel("Mean Squared Error")
plt.title("GBM Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("gbm_training_vs_validation_loss.png", dpi=300)
plt.show()

print("Saved: gbm_training_vs_validation_loss.png")
#------------------------------------------------------
#16. Parity Plots for GBM 
#------------------------------------------------------
# GBM predictions
y_all = np.concatenate([y_train, y_test])
ymin, ymax = y_all.min(), y_all.max()

def parity_plot(y_true, y_pred, model_name, split_name, filename, ymin, ymax):
    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=14, color="tab:blue")

    # 1:1 reference line
    plt.plot([ymin, ymax], [ymin, ymax], 'k--', linewidth=2)

    plt.xlim(ymin, ymax)
    plt.ylim(ymin, ymax)

    plt.xlabel("True CO₂ Loading (mol/kg)", fontsize=12)
    plt.ylabel("Predicted CO₂ Loading (mol/kg)", fontsize=12)

    # Professional title
    plt.title(f"{model_name} Parity Plot ({split_name} Set)", fontsize=13)

    # Metrics box (top-right)
    metrics_text = (
        f"$R^2$ = {r2:.3f}\n"
        f"RMSE = {rmse:.2f} mol/kg"
    )

    plt.text(
        0.97, 0.03, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

parity_plot(
    y_train, gbm_train_pred,
    model_name="Gradient Boosting (GBM)",
    split_name="Training",
    filename="gbm_parity_train.png",
    ymin=ymin, ymax=ymax
)

parity_plot(
    y_test, gbm_test_pred,
    model_name="Gradient Boosting (GBM)",
    split_name="Test",
    filename="gbm_parity_test.png",
    ymin=ymin, ymax=ymax
)	



