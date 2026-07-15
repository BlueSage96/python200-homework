import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Task 1
students_data = pd.read_csv("student_performance_math.csv",sep=";")
students_df = pd.DataFrame(students_data)
g3 = students_df["G3"] #cluster of 0s are the students who didn't take the final exam
fig, ax = plt.subplots()

print(f"\nShape: {students_df.shape}")
print(f"\nFive rows:\n {students_df.head(5)}")
print(f"\nData Types:\n {students_df.dtypes}")

plt.hist(g3,bins=21,color="green")
plt.xlabel("Final Grade")
plt.ylabel("Grade Prediction")
plt.title("Distribution of Final Math Grades")
plt.savefig("outputs/g3_distributon.png")
plt.show()

# Task 2
print(f"\nBefore G3 0: {students_df.shape}")
# Filter out 0s & save to new dataframe
g30 = students_df[(students_df["G3"] == 0)]
# Drop 0s
students_df2 = students_df.drop(g30.index)
print(f"\nAfter G3 0:\n {students_df2.shape}")

#Removing 0s helps simiplify the dataset before 
# converting yes/no to 1/0 and sex column 0/1
#There will be less confusion with these conversions.

students_df2[["schoolsup","internet","higher","activities"]] = students_df2[["schoolsup","internet","higher","activities"]].apply({lambda x: 0 if x == 'no' else 1})
students_df2[["sex"]] = students_df2[["sex"]].apply({lambda x: 0 if x == 'F' else 1})

pearson1 = pearsonr(students_df2["absences"],students_df2["G3"])
print(f"\nOriginal dataset:\n {pearson1}")

pearson2 = pearsonr(students_df2["absences"],students_df2["G3"])
print(f"\nUpdated dataset:\n {pearson2}")
#Filtering out the G3 0's caused the number of abscenses to decline - 
# the students who didn't take the final exam were abscent in the original dataset.

# Task 3
g3 = students_df2["G3"]
sorted_pearson = []
numeric_cols = students_df2[["sex","age","Medu","Fedu","traveltime","studytime",
                            "failures","schoolsup","internet","higher","activities",
                            "freetime","goout","Walc","G1","G2","G3"]]
for cols in numeric_cols:
    if cols != "G3":
       pearson1 = pearsonr(students_df2[cols],g3)
       sorted_pearson.append((cols,pearson1)) #creates a tuple: ((a,b))
       sorted_pearson.sort()
print("\nPearson comparison:\n\n",sorted_pearson)
       
#"traveltime" has the strongest relationship with G3. 
# I'm surpised "Fedu" and "Medu" are some of the weaker statistics. 
# I thought the parent's education would have a positive 
# influence on the students' education.

plt.figure(figsize=(10,8))
students_corr = students_df2.corr()
sns.heatmap(students_corr,cmap="coolwarm")
plt.title("students Heatmap")
plt.savefig("outputs/students_heatmap.png")
plt.show()

#G1 has a "positive" (0.6-0.75) relationship with both G2 and G3
#G1-G3 has the weakest relationships with Medu, Fedu, and studytime
#Failures and Age hardly any relationship with one another.
categories = students_df2.groupby("G3")["G1"].mean()

plt.bar(categories.index,categories.values,color=["Blue","Red"])
plt.title("Students Bar Plot")
plt.xlabel("G3")
plt.ylabel("Total G1")
plt.savefig("outputs/g1_g3_bar_graph")
plt.show()

#There's a high corrleation for the students in first period & third period

# Task 4
model = LinearRegression()
failures = students_df2["failures"]
failures = failures.to_numpy()
X = failures.reshape(-1,1)
y = g3

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nSlope:",model.coef_[0])
print(f"RMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"R2:",model.score(X_test, y_test))

# G3 students didn't score well on the final exam based on the slope and RMSE
# R2 did very little in increasing the slope and RMSE

# Task 5
feature_cols = ["failures","Medu","Fedu","studytime","higher","schoolsup",
                "internet","sex","freetime","activities","traveltime"]
X = students_df2[feature_cols].values
y = students_df2["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nRMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"Train R2:",model.score(X_train, y_train))
print(f"Test R2:\n",model.score(X_test, y_test))

# The test R2 helps a little, meaning that adding features had a lukewarm effect on task 4's R2

# Print each feature name & its cofficient
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}:{coef:+3f}")
# No surprise and the R2 are close, and that tells me the model has an overall weak relationship
# I would drop failures, schoolsup, and traveltime as they cause a lot of the weak model relationship.
# Activities is fine since the number is lower than the above.

# Task 6
plt.plot( [0,20],[0,20], color="black")
plt.scatter(y_pred,y_test,color="green",cmap="coolwarm")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual (Full Model)")
plt.savefig("outputs/predicted_vs_actual_full.png")
plt.show()

# 1. The size of the filtered dataset and the test set
# After filtered dataset contained 357 students and 71-72 for the test set.

# 2. The RMSE and R² of your best model in plain language -- 
# on a 0-20 scale, what does a typical prediction error actually mean?
# The lower the RMSE is to 0 or 1 the more accurate the prediction is.
# Since the R2 is 0.15-0.17, it can only determine the 15-17% of student's final grades.

# 3. Which two features have the largest positive and largest negative coefficients, and what those mean?
#  The largest positive feature is internet which makes sense given that mostly kid & teens use it.
#  The largest negative feature is schoolsup meaning that students didn't use the extra educational support.

# 4. One result that surprised you
# The biggest surprise is schoolsup because it would make sense for students to take advantage 
# of the extra resources from the school


# Extra: G1
feature_cols = ["failures","Medu","Fedu","studytime","higher","schoolsup",
                "internet","sex","freetime","activities","traveltime","G1"]
X = students_df2[feature_cols].values
y = students_df2["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nRMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"Train R2:",model.score(X_train, y_train))
print(f"Test R2:",model.score(X_test, y_test))

# 1. Does a high R² here mean G1 is causing G3? 
# The high R2 shows a strong correlation between G1 and G3.
# However, the high correlation does not mean that G1 has any influence on G3.

# 2. Is this a useful model for identifying students who might struggle? 
# Yes as long as G1 is a feature because the period grades are sequential 
# and G1 and G2 cannot be skipped.

# 3. What might educators need to do if they wanted to intervene early,
# before G1 is even available?
# Even without G1, teachers can encourage students to take advantage of 
# schoolsup, studytime and decrease absences which leads to less failures.


