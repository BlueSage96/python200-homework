import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Task 1
student_data = pd.read_csv("student_performance_math.csv",sep=";")
student_df = pd.DataFrame(student_data)
g3 = student_df["G3"] #cluster of 0s are the students who didn't take the final exam

print(f"Shape: {student_df.shape}")
print(f"Five rows:\n {student_df.head(5)}")
print(f"Data Types:\n {student_df.dtypes}")

plt.hist(g3,bins=21,color="green")
plt.xlabel("Final Grade")
plt.ylabel("Grade Prediction")
plt.title("Distribution of Final Math Grades")
plt.savefig("outputs/g3_distributon.png")
plt.show()

# Task 2
print(f"Before G3 0: {student_df.shape}")
# Filter out 0s & save to new dataframe
g30 = student_df[(student_df["G3"] == 0)]
# Drop 0s
student_df2 = student_df.drop(g30.index)
print(f"After G3 0: {student_df.shape}")

#Removing 0s helps simiplify the dataset before 
# converting yes/no to 1/0 and sex column 0/1
#There will be less confusion with these conversions.

student_df2[["schoolsup","internet","higher","activities"]] = student_df2[["schoolsup","internet","higher","activities"]].apply({lambda x: 0 if x == 'no' else 1})
student_df2[["sex"]] = student_df2[["sex"]].apply({lambda x: 0 if x == 'F' else 1})

pearson1 = pearsonr(student_df["absences"],student_df["G3"])
print(f"Original dataset:\n {pearson1}")

pearson2 = pearsonr(student_df2["absences"],student_df2["G3"])
print(f"Updated dataset:\n {pearson2}")
#Filtering out the G3 0's caused the number of abscenses to decline - 
# the students who didn't take the final exam were abscent in the original dataset.