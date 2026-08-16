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

# The CSV uses semicolons (;) as the separator between columns.
students_data = pd.read_csv("student_performance_math.csv",sep=";")
students_df = pd.DataFrame(students_data)
g3 = students_df["G3"] #cluster of 0s are the students who didn't take the final exam
fig, ax = plt.subplots()

print(f"\nShape: {students_df.shape}")
print(f"\nFive rows:\n {students_df.head(5)}")
print(f"\nData Types:\n {students_df.dtypes}")

plt.hist(students_df["G3"], bins=21)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Math Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("outputs/g3_distribution.png")
plt.show()

# Task 2
print(f"\nTask 2:\n")
print(f"\nBefore G3 0: {students_df.shape}")
# Filter out 0s & save to new dataframe
g30 = students_df[(students_df["G3"] == 0)]
# Drop 0s
students_df2 = students_df.drop(g30.index)
print(f"\nAfter G3 0:\n {students_df2.shape}")

'''
Removing 0s helps simiplify the dataset before 
converting yes/no to 1/0 and sex column 0/1
There will be less confusion with these conversions.
'''

students_df2[["schoolsup","internet","higher","activities"]] = students_df2[["schoolsup","internet","higher","activities"]].apply(
    lambda col: col.map({"yes": 1, "no": 0})
)
students_df2["sex"] = students_df2["sex"].map({"F": 0, "M": 1})

pearson1 = pearsonr(students_df["absences"],students_df["G3"])
print(f"\nOriginal dataset:\n {pearson1}")

pearson2 = pearsonr(students_df2["absences"],students_df2["G3"])
print(f"\nUpdated dataset:\n {pearson2}")

'''
G3 = 0 represents students who did not receive a final exam grade.
Keeping these rows could distort the regression because a 0 does not
represent a normal final grade. Removing these rows allows the model
to learn from students who actually received a final grade.
'''

# Task 3
print(f"\nTask 3:\n")

g3 = students_df2["G3"]
sorted_pearson = []
numeric_cols = students_df2[[
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "freetime",
    "goout",
    "Walc",
    "absences"
]]
for cols in numeric_cols:
    if cols != "G3":
       pearson1 = pearsonr(students_df2[cols],g3)
       sorted_pearson.append((cols,pearson1)) #creates a tuple: ((a,b))
       
# Sort by Pearson corrleation coefficient
sorted_pearson.sort(key=lambda x: x[1].statistic)
print("\nPearson comparison:\n")

for feature,result in sorted_pearson:
    print(f"{feature:12s}{result.statistic:+3f}")
      
'''
Failures has the strongest negative relationship with G3, with a Pearson
correlation of about -0.294. Medu has the strongest positive relationship
with G3 among the selected numeric features, with a correlation of about
0.190. Most of the selected features have relatively weak relationships
with G3. I was surprised that parental education had only a weak positive
relationship with final grade.
'''

plt.figure(figsize=(10,8))
students_corr = students_df2.corr()
sns.heatmap(students_corr,cmap="coolwarm")
plt.title("students Heatmap")
plt.savefig("outputs/students_heatmap.png")
plt.show()

'''
The heatmap shows that G1 and G2 have the strongest relationships with G3,
while most of the other features have weaker relationships with final grade.
'''

# Pearson correlation bar chart
features = [item[0] for item in sorted_pearson]
correlations = [item[1].statistic for item in sorted_pearson]

plt.figure(figsize=(10,8))
plt.barh(features, correlations)
plt.xlabel("Pearson Correlation with G3")
plt.ylabel("Feature")
plt.title("Feature Correlations with G3")
plt.axvline(0)
plt.savefig("outputs/pearson_correlations.png")
plt.show()

'''
The bar chart shows that G1 and G2 have the strongest positive
relationships with G3, while failures has the strongest negative
relationship. Most of the other features have relatively weak
relationships with final grade.
'''

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

print(f"\nTask 4:\n")
print(f"\nSlope:",model.coef_[0])
print(f"RMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"R2:",model.score(X_test, y_test))

'''
The failures feature has a negative relationship with G3, meaning students
with more past failures tend to have lower final grades. The R2 is low, so
failures by itself does not explain much of the variation in final grades.
'''

# Task 5
feature_cols = [
    "sex",
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "freetime",
    "goout",
    "Walc",
    "absences"
]
X = students_df2[feature_cols].values
y = students_df2["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nTask 5:\n")
print(f"\nRMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"Train R2:",model.score(X_train, y_train))
print(f"Test R2:\n",model.score(X_test, y_test))

# Print each feature name & its cofficient
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name} coefficient: {coef}")
'''
The full model has a test R2 of about 0.26, meaning it explains about
26% of the variation in students' final grades. The RMSE is about 2.67,
so the model's predictions are typically about 2.67 grade points away
from the actual G3 score on a 0-20 scale.

The largest positive coefficient is internet at about +1.09. Holding
the other features constant, internet access is associated with a
higher predicted G3.

The largest negative coefficient is schoolsup at about -2.13. Holding
the other features constant, receiving school support is associated
with a lower predicted G3. This does not mean school support causes
lower grades; students who are already struggling may be more likely
to receive school support.

I was surprised that school support had the largest negative coefficient,
while internet access had the largest positive coefficient.
'''

# Task 6
plt.plot( [0,20],[0,20], color="black")
plt.scatter(y_pred,y_test,color="green")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual (Full Model)")

plt.savefig("outputs/predicted_vs_actual.png")
# This overwrites the Warmup Q5 plot because both assignments require this filename.
plt.show()

'''
Task 6 Summary:

Dataset size: The filtered dataset contains 357 students, and the test
set contains 72 students.

Model error: The RMSE is about 2.67. Since G3 is scored from 0 to 20,
the model's predictions are typically about 2.67 grade points away from
the actual final grade.

Model R2: The test R2 is about 0.26, meaning the model explains about
26% of the variation in students' final grades.

Largest positive coefficient: Internet has the largest positive
coefficient at about +1.09. Holding the other features constant,
internet access is associated with a higher predicted G3.

Largest negative coefficient: Schoolsup has the largest negative
coefficient at about -2.13. Holding the other features constant,
school support is associated with a lower predicted G3. This does not
mean school support causes lower grades; students who are already
struggling may be more likely to receive school support.

Surprising result: I was surprised that school support had the largest
negative coefficient, since school support is intended to help students.
The result may reflect that students who need more support are already
more likely to be struggling academically.
'''

# Extra: G1
feature_cols = ["failures","Medu","Fedu","studytime","higher","schoolsup",
                "internet","sex","freetime","activities","traveltime", 
                "absences","goout","Walc","G1"]
X = students_df2[feature_cols].values
y = students_df2["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nExtra G1:\n")
print(f"\nRMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"Train R2:",model.score(X_train, y_train))
print(f"Test R2:",model.score(X_test, y_test))

'''
Adding G1 greatly increases the model's test R2 from about 0.26 to about
0.75, showing that G1 is a very strong predictor of G3. The RMSE is about
1.54, meaning the model's predictions are typically about 1.54 grade
points away from the actual G3 score on a 0-20 scale.

However, a high R2 does not mean that G1 causes G3. The two grades are
strongly related because they measure student performance at different
points during the school year.

This model could be useful for identifying students who may struggle
with G3 once G1 is available, allowing educators to provide additional
support.

If educators want to intervene before G1 is available, they would need
to use information available earlier, such as previous academic
performance, attendance, past failures, study habits, and other student
support factors. This could help identify students who may need support
before their first-period grade is available.
'''