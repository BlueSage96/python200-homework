import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

#Task 01
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
df.head()

spam = df[df["spam_label"] == 1]
ham = df[df["spam_label"] == 0]

print(f"\nTask 01:\n")
print(f"Total emails: {len(df)}")
print(f"\nClass counts:\n{df['spam_label'].value_counts()}")
print(f"\nClass percentages:\n{df['spam_label'].value_counts(normalize=True) * 100}")


fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].boxplot(x=[spam["word_freq_free"],ham["word_freq_free"]],
            labels=["Spam","Ham"],patch_artist=True,medianprops={'color':'red'})
ax[0].set_title("Word Frequency Free")

ax[1].boxplot(x=[spam["char_freq_!"],ham["char_freq_!"]],
            labels=["Spam","Ham"],patch_artist=True,medianprops={'color':'white'})
ax[1].set_title("Character Frequency")

ax[2].boxplot(x=[spam["capital_run_length_total"],ham["capital_run_length_total"]],
            labels=["Spam", "Ham"],patch_artist=True,medianprops={'color':'blue'})
ax[2].set_title("Capital Length Total")
plt.savefig("outputs/spam_ham_comparisons.png")
plt.show()


# 1. Spam and ham emails have some differences in these features.
#    Some of the differences are more noticeable than others.

# 2. A lot of the values are 0 because many emails do not use
#    certain words or characters at all.

# 3. Some features have very small values while others have much
#    larger values, so the feature scales are very different.

# 4. There are more ham emails than spam emails, so the classes are
#    not evenly balanced. Accuracy alone may not show how well the model
#    identifies spam.

#Task 02

# Scale the training data first because the features use very different
# numeric ranges. Fit both the scaler and PCA on training data only so
# information from the test set does not leak into training.

#Remove spam_label for X data 
X = df.drop("spam_label",axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # use the mean and standard deviation learned from training data

pca = PCA()
pca_fit = pca.fit(X_train_scaled)

perc_exp_vals = pca_fit.explained_variance_ratio_ 
total_explained = np.cumsum(perc_exp_vals)

n = 0

for i, value in enumerate(total_explained):
    if value >= 0.90:
        n = i + 1
        break

plt.plot(range(1, len(total_explained) + 1), total_explained, color="turquoise")
plt.title("PCA Variance Explained")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("outputs/pca_variance_project_03.png")
plt.show()    

print(f"\nTask 02:\n")
print("Explained variance:", ", ".join(f"{v:.2f}" for v in perc_exp_vals))
print(f"Components needed for 90% explained variance: {n}")

#Transform both train and test data and slice the first n components
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

#Task 03
print(f"\nTask 03:\n")

#KNN on unscaled data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
score = accuracy_score(y_test,preds)
class_report = classification_report(y_test,preds)

print(f"\nKNN 01:\n")
print(f"Accuracy: {score}")
print(f"Report:\n {class_report}")

#KNN on scaled data
knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled,y_train)
preds2 = knn2.predict(X_test_scaled)
score2 = accuracy_score(y_test,preds2)
class_report2 = classification_report(y_test,preds2)

print(f"\nKNN 02:\n")
print(f"Accuracy: {score2}")
print(f"Report:\n {class_report2}")

#KNN with PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
knn_pca_preds = knn_pca.predict(X_test_pca)
class_report3 = classification_report(y_test, knn_pca_preds)

print(f"\nKNN 03:\n")
print(f"Accuracy: {accuracy_score(y_test, knn_pca_preds)}\n")
print(f"Report:\n {class_report3}")

#Decision Tree 01
dtc1 = DecisionTreeClassifier(max_depth=3,random_state=42)
dtc1.fit(X_train,y_train)
train_preds_dtc1 = dtc1.predict(X_train)
test_preds_dtc1 = dtc1.predict(X_test)

train_accuracy_dtc1 = accuracy_score(y_train, train_preds_dtc1)
test_accuracy_dtc1 = accuracy_score(y_test,test_preds_dtc1)

print(f"\nDecision Tree 01:\n")
print(f"Train Accuracy: {train_accuracy_dtc1}")
print(f"Test Accuracy: {test_accuracy_dtc1}")

#Decision Tree 02
dtc2 = DecisionTreeClassifier(max_depth=5,random_state=42)
dtc2.fit(X_train,y_train)
train_preds_dtc2 = dtc2.predict(X_train)
test_preds_dtc2 = dtc2.predict(X_test)

train_accuracy_dtc2 = accuracy_score(y_train, train_preds_dtc2)
test_accuracy_dtc2 = accuracy_score(y_test,test_preds_dtc2)

print(f"\nDecision Tree 02:\n")
print(f"Train Accuracy: {train_accuracy_dtc2}")
print(f"Test Accuracy: {test_accuracy_dtc2}")

#Decision Tree 03
dtc3 = DecisionTreeClassifier(max_depth=10,random_state=42)
dtc3.fit(X_train,y_train)
train_preds_dtc3 = dtc3.predict(X_train)
test_preds_dtc3 = dtc3.predict(X_test)

train_accuracy_dtc3 = accuracy_score(y_train, train_preds_dtc3)
test_accuracy_dtc3 = accuracy_score(y_test,test_preds_dtc3)
class_report_dtc3 = classification_report(y_test, test_preds_dtc3)

print(f"\nDecision Tree 03:\n")
print(f"Train Accuracy: {train_accuracy_dtc3}")
print(f"Test Accuracy: {test_accuracy_dtc3}")

#Decision Tree 04
dtc4 = DecisionTreeClassifier(max_depth=None,random_state=42)
dtc4.fit(X_train,y_train)
train_preds_dtc4 = dtc4.predict(X_train)
test_preds_dtc4 = dtc4.predict(X_test)

train_accuracy_dtc4 = accuracy_score(y_train, train_preds_dtc4)
test_accuracy_dtc4 = accuracy_score(y_test,test_preds_dtc4)

print(f"\nDecision Tree 04:\n")
print(f"Train Accuracy: {train_accuracy_dtc4}")
print(f"Test Accuracy: {test_accuracy_dtc4}")

# 1. As the tree depth increases, the training accuracy keeps increasing.
# The gap between the training and test accuracy also gets larger, which shows overfitting.

# 2. I chose Decision Tree 03 with max_depth=10 because its test accuracy
#    is close to the unlimited tree, but the train/test gap is smaller,
#    so it shows less overfitting.

print(f"\nDecision Tree 03 accuracy and report:\n")
print(f"Test Accuracy: {test_accuracy_dtc3}")
print(f"Report:\n {class_report_dtc3}")

#Random Foreset Classifier
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

rf_score = accuracy_score(y_test, rf_pred)
rf_class_report = classification_report(y_test, rf_pred)

print(f"\nRandom Forest 01:\n")
print(f"Accuracy: {rf_score}")
print(f"Report:\n {rf_class_report}")


tree_importance = pd.Series(
    dtc3.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nDecision Tree Top 10 Features:")
print(tree_importance.head(10))

print("\nRandom Forest Top 10 Features:")
print(rf_importance.head(10))

rf_importance.head(10).sort_values().plot(kind="barh")

plt.title("Random Forest Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.show()

# The Decision Tree and Random Forest agree on several important features,
# including char_freq_!, char_freq_$, word_freq_remove, word_freq_free,
# capital_run_length_total, and word_freq_hp. This makes sense because
# things like special characters, certain words, and capital letters
# could help distinguish spam from ham emails.

#Logistic Regression
logistic_scaled = LogisticRegression(C=1.0,max_iter=1000,solver="liblinear")
logistic_pca = LogisticRegression(C=1.0,max_iter=1000,solver="liblinear")

logistic_scaled.fit(X_train_scaled,y_train)
logistic_pca.fit(X_train_pca,y_train)

scaled_preds = logistic_scaled.predict(X_test_scaled)
pca_preds = logistic_pca.predict(X_test_pca)

logistic_scaled_report = classification_report(y_test, scaled_preds)
logistic_pca_report = classification_report(y_test, pca_preds)

print(f"\nLogical Regression 01:\n")

print("\nLogistic Regression Scaled:")
print(f"Accuracy: {accuracy_score(y_test, scaled_preds)}\n")
print(f"Report:\n{logistic_scaled_report}\n")

print("\nLogistic Regression PCA:")
print(f"Accuracy: {accuracy_score(y_test, pca_preds)}\n")
print(f"Report:\n{logistic_pca_report}")

# PCA did not improve Logistic Regression. The scaled version had
# 92.94% accuracy while the PCA version had 91.86% accuracy.

# Random Forest performed the best with an accuracy of about 94.57%.

# I would prioritize reducing false positives because I would rather
# have some spam get through than have an important email marked as spam.

cm = confusion_matrix(y_test,rf_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("RandomForestClassifier Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png")

# The Random Forest has more false negatives (32) than false positives (18).
# This means it is more likely to let spam into the inbox than mark a ham email as spam.

#Task 04
print(f"\nTask 04:\n")
cv_scores_knn1 = cross_val_score(knn, X_train, y_train, cv=5)

print(f"\nKNN 01:\n")
print(f"Mean fold scores: {cv_scores_knn1.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_knn1.std():.3f}")

cv_scores_knn2 = cross_val_score(knn2, X_train_scaled, y_train, cv=5)

print(f"\nKNN 02:\n")
print(f"Mean fold scores: {cv_scores_knn2.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_knn2.std():.3f}")

cv_scores_knn3 = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)

print(f"\nKNN 03:\n")
print(f"Mean fold scores: {cv_scores_knn3.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_knn3.std():.3f}")

cv_scores_dtc1 = cross_val_score(dtc1, X_train, y_train, cv=5)

print(f"\nDecision Tree 01:\n")
print(f"Mean fold scores: {cv_scores_dtc1.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_dtc1.std():.3f}")

cv_scores_dtc2 = cross_val_score(dtc2, X_train, y_train, cv=5)

print(f"\nDecision Tree 02:\n")
print(f"Mean fold scores: {cv_scores_dtc2.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_dtc2.std():.3f}")

cv_scores_dtc3 = cross_val_score(dtc3, X_train, y_train, cv=5)

print(f"\nDecision Tree 03:\n")
print(f"Mean fold scores: {cv_scores_dtc3.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_dtc3.std():.3f}")

cv_scores_dtc4 = cross_val_score(dtc4, X_train, y_train, cv=5)

print(f"\nDecision Tree 04:\n")
print(f"Mean fold scores: {cv_scores_dtc4.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_dtc4.std():.3f}")

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)

print(f"\nRandom Forest 01:\n")
print(f"Mean fold scores: {cv_scores_rf.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_rf.std():.3f}")

cv_scores_logistic_scaled = cross_val_score(logistic_scaled, X_train_scaled, y_train, cv=5)

print(f"\nLogistic Regression 01:\n")
print(f"Mean fold scores: {cv_scores_logistic_scaled.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_logistic_scaled.std():.3f}")

cv_scores_logistic_pca = cross_val_score(logistic_pca, X_train_pca, y_train, cv=5)

print(f"\nLogistic Regression 02:\n")
print(f"Mean fold scores: {cv_scores_logistic_pca.mean():.3f}")
print(f"Standard deviation of fold scores: {cv_scores_logistic_pca.std():.3f}")

# Random Forest had the highest mean cross-validation accuracy at 0.954,
# so it was the most accurate model.

# Logistic Regression with PCA had the lowest standard deviation at 0.003,
# so it was the most stable model.

# Random Forest also performed best on the single train/test split,
# so the cross-validation results support the earlier ranking.

# Task 05

rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100,random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pipeline_pred = rf_pipeline.predict(X_test)
rf_pipeline_report = classification_report(y_test, rf_pipeline_pred)

lr_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
lr_pipeline_pred = lr_pipeline.predict(X_test)
lr_pipeline_report = classification_report(y_test, lr_pipeline_pred)

print(f"\nTask 05:\n")
print(f"RandomForestClassifier Report:\n {rf_pipeline_report}\n")
print(f"LogisticRegression Report:\n {lr_pipeline_report}\n")

# The pipelines do not have the same structure because Logistic Regression
# benefits from scaling while Random Forest does not need scaled data.
# Pipelines keep preprocessing and the model together, which makes it easier
# to apply the same steps consistently when the model is reused or deployed.