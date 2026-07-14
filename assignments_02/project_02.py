import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

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
print(f"\nAfter G3 0:\n {students_df.shape}")

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
                            "failures","schoolsup","internet","higher","activities","freetime","goout","Walc","G1","G2","G3"]]
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