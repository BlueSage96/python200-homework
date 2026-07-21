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
# Using X_train because the X_train contains the mean and std for only the training data.

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
# The accuracy is the same for unscaled and scaled data, so scaling makes no difference

#KNN 03
knn3 = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn3,X_train,y_train,cv=5)

print(f"\nKNN 03:\n")
print(f"Fold scores: {cv_scores}")
print(f"Mean fold scores: {cv_scores.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores.std():.3f}")

# The cross_val_score is more trustworthy than a single train/test 
# split because each group of training data (fold) is evaluated 
# and the average score is more stable than any single split.

#KNN 04
print(f"\nKNN 04:\n")

k_values = [1,3,5,7,9,11,13,15]
for k in k_values:
    knn4 = KNeighborsClassifier(n_neighbors=5)
    cross = cross_val_score(knn4,X_train,y_train,cv=5)
    print(f"k={k:2d}: mean={cross.mean():.3f}")
    # I would use k=15 as it has a wider range of neighbors to average than the other ks. \z
    
    
#Classifier Evaluation 01
cm = confusion_matrix(y_test,preds)
display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)
display.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()

#Decision Tree 01
dtc = DecisionTreeClassifier(max_depth=3,random_state=42)
dtc.fit(X_train,y_train)
preds3 = dtc.predict(X_test)
score3 = accuracy_score(y_test, preds3)
class_report2 = classification_report(y_test, preds3)

print(f"\nDecision Trees 01:\n")
print("Accuracy:", score3)
print(class_report2)
# KNN's accuracy is 1.0 while Decision Tree accuracy is 0.9666666666666667
# No scaled vs. unscaled data would not affect the result.

#Logical Regression 01
log_reg1 = LogisticRegression(C=0.01,max_iter=1000,solver='liblinear')
log_reg2 = LogisticRegression(C=1.0,max_iter=1000,solver='liblinear')
log_reg3 = LogisticRegression(C=100,max_iter=1000,solver='liblinear')

log_reg1.fit(X_train_scaled,y_train)
log_reg2.fit(X_train_scaled,y_train)
log_reg3.fit(X_train_scaled,y_train)

log_reg_np1 = np.abs(log_reg1.coef_).sum()
log_reg_np2 = np.abs(log_reg2.coef_).sum()
log_reg_np3 = np.abs(log_reg3.coef_).sum()

print(f"\nLogical Regression 01:\n")
print(f"Model one C value: {log_reg1.C} and total size: {log_reg_np1}")
print(f"Model two C value: {log_reg2.C} and total size: {log_reg_np2}")
print(f"Model three C value: {log_reg3.C} and total size: {log_reg_np3}")
# The coefficients increase significally depending on the size of C.

# PCA
digits = load_digits()
X_digits = digits.data # 1797 images, each flattened to 64 pixel values
y_digits = digits.target # digit labels 0-9
images = digits.images # same data shaped as 8x8 images for plotting

#PCA 01
print(f"\nPCA 01:\n")
print(f"Shape of x digits: {X_digits.shape}")
print(f"Shape of images: {images.shape}")

fig, ax = plt.subplots(1, 10, figsize=(15, 2))
#use a loop to prevent "repetitive code"!
for i in range(10):
    ax[i].imshow(images[i],cmap="gray_r")
    ax[i].set_title(y_digits[i])
    ax[i].axis("off")
plt.savefig("outputs/sample_digits.png")
plt.show()

#PCA 02
fig, ax1 = plt.subplots()
pca = PCA(svd_solver="randomized",random_state=0)
pca_fit = pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = ax1.scatter(scores[:,0],scores[:,1],c=y_digits,cmap="tab10",s=10) # c = color array
plt.colorbar(scatter,label="Digit")
plt.title("PCA 2D Projection")
plt.savefig("outputs/pca-2d_projection.png")
plt.show()

#PCA 03
perc_exp_vals = pca_fit.explained_variance_ratio_ 
total_explained = np.cumsum(perc_exp_vals)

plt.scatter(perc_exp_vals,total_explained,color="orange")
plt.title("PCA Variance Explained")

plt.xlabel("Exp Vals")
plt.ylabel("Total Explained")

plt.savefig("outputs/pca_variance_explained.png")
plt.show()
#The components are very small and would take 13 components to explain 80% of variance

#PCA 04
def reconstruct_digit(sample_idx,scores,pca,n_components):
    #Reconstruct one digit using the first n_components principal components
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx,i] * pca.components_[i]
    return reconstruction.reshape(8,8)


fig, axs = plt.subplots(5,5,figsize=(10,10))
component_counts = [2, 5, 15, 40]

for row, n in enumerate(component_counts, start=1):
    for col in range(5):
        axs[0][col].imshow(images[col], cmap="gray")
        reconstruction = reconstruct_digit(col, scores, pca, n)
        axs[row][col].imshow(reconstruction, cmap="gray")
plt.savefig("outputs/pca_reconstructions.png")
plt.show()
#Numbers become recognizable at n=40 and it matches where the variance curve levels off.