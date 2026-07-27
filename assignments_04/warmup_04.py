import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scaled data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#ROC 01
print(f"ROC 01:\n")
log_reg = LogisticRegression(max_iter=1000,random_state=42)
log_reg.fit(X_train,y_train)
y_probs = log_reg.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_probs)

print(f"AUC not scaled: {auc:.3f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
knn_probs = knn.predict_proba(X_test_scaled)[:,1]
knn_auc = roc_auc_score(y_test,knn_probs)

print(f"AUC scaled: {knn_auc:.3f}")

#KNN has the better auc score meaning it has a better-discriminating 
# model than the logistic regression.

#ROC 02
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test,knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr,tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC={auc:.2f})")
RocCurveDisplay(fpr=knn_fpr, tpr=knn_tpr).plot(ax=ax, name=f"KNN k=5 (AUC={knn_auc:.2f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("KNN and Linear Regression ROC Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.show()

#1. KNN has the lower FPR when each model reaches TPS = 0.80.
#2. KNN would produce fewer alarms when needing to catch 80% of positives.

#ROC 03
best_f1 = 0
# use enumerate and get indices/positions
for i,t in enumerate(thresholds):
    y_pred = (y_probs >= t).astype(int)
    f1 = f1_score(y_test,y_pred)
    if f1 > best_f1:
        # Save values
        best_f1 = f1
        best_threshold = t
        best_tpr = tpr[i]
        best_fpr = fpr[i]
       
print(f"\nROC 03:\n")
print(f"TPR: {best_tpr}")
print(f"FPR: {best_fpr}")
print(f"Thresholds: {best_threshold}")
print(f"F1: {best_f1}")

#1. The optimal threshold is below 0.5 (it's 0.28)
#2. In a real world application, I would choose a 
# lower threshold than 0.5 to make sure I get the optimal threshold

#GridSearch 01
pipe = Pipeline([("scaler",StandardScaler(),("clf",LogisticRegression(max_iter=1000)))])
param_grid = {
    "C":[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000,random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train_scaled,y_train)

best_lr = grid_search.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_probs_lr = best_lr.predict_proba(X_test_scaled)[:,1]
auc_lr = roc_auc_score(y_test,y_pred_lr)

print(f"\nGridSearch 01:\n")
print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print(f"Test AUC: {auc_lr}")

#1. I guessed 10.0 by default, not the actual best C (100.0)
#2. The AUC changed by 100%
