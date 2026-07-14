import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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