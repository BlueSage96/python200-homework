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
print(f"X-train Shape: {X_train.shape}")
print(f"X-test Shape: {X_test.shape}")
print(f"y-train Shape: {y_train.shape}")
print(f"y-test Shape: {y_test.shape}")