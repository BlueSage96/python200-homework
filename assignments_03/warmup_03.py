import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#Preprocessiong Q1
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, stratify=y,random_state=42
)
print(f"Preprocessing 01:\n")
print(f"X-train Shape: {X_train.shape}\n")
print(f"X-test Shape: {X_test.shape}\n")
print(f"y-train Shape: {y_train.shape}\n")
print(f"y-test Shape: {y_test.shape}\n")

#Preprocessing 02
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"\nPreprocessing 02:\n")
print("Mean of sepal length:", X_train_scaled[0].mean())
print("Mean of sepal width:",X_train_scaled[1].mean())
print("Mean of petal length:",X_train_scaled[2].mean())
print("Mean of petal width:",X_train_scaled[3].mean())
#Using X_train because the X_train contains the mean and std for only the training data.

#KNN 01
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)

score = accuracy_score(y_test, preds)
class_report = classification_report(y_test, preds)

print(f"\nKNN 01:\n")
print("Accuracy:", score)
print(class_report)