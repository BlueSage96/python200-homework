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
print("Mean of sepal length:", X_train_scaled[:,0].mean())
print("Mean of sepal width:",X_train_scaled[:,1].mean())
print("Mean of petal length:",X_train_scaled[:,2].mean())
print("Mean of petal width:",X_train_scaled[:,3].mean())
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

#KNN 02
knn2 = KNeighborsClassifier(n_neighbors=5)
X_test_scaled = scaler.fit_transform(X_test)
knn2.fit(X_train_scaled,y_train) #Xs - features y's - labels

preds2 = knn2.predict(X_test_scaled)
score2 = accuracy_score(y_test, preds2)

print(f"\nKNN 02:\n")
print("Accuracy:", score2)
#The accuracy is the same for unscaled and scaled data, so scaling makes no difference

#KNN 03
knn3 = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn3,X_train,y_train,cv=5)

print(f"\nKNN 03:\n")
print(f"Fold scores: {cv_scores}")
print(f"Mean fold scores: {cv_scores.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores.std():.3f}")

#The cross_val_score is more trustworthy than a single train/test 
# split because each group of training data (fold) is evaluated 
# and the average score is more stable than any single split.

#KNN 04
print(f"\nKNN 04:\n")

k_values = [1,3,5,7,9,11,13,15]
for k in k_values:
    knn4 = KNeighborsClassifier(n_neighbors=5)
    cross = cross_val_score(knn4,X_train,y_train,cv=5)
    print(f"k={k:2d}: mean={cross.mean():.3f}")
    #I would use k=15 as it has a wider range of neighbors to average than the other ks. 