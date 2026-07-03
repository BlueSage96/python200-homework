import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Pandas 01
data = {
    "name": ["Alice","Bob","Carol","David","Eve"],
    "grade": [85,72,90,68,95],
    "city": ["Boston","Austin","Boston","Denver","Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
print(f"Pandas Q1:")
print(f"First three rows: {df.head(3)}")
print(f"Shape:{df.shape}")
print(f"Column data types: {df.dtypes}")

# Pandas 02
df1 = df[(df['grade'] > 80)]
print(f"Pandas Q2:\n {df1}")

# Pandas 03
df2 = df.copy()
df2["grade_curved"] = df[["grade"]].apply(lambda x: x["grade"] + 5,axis=1)
print(f"Pandas Q3:\n {df2}")

# Pandas 04
df2["name_upper"] = df2[["name"]]
df2["name_upper"] = df2["name_upper"].str.upper()
print(f"Pandas Q4:")
print(df2[["name","name_upper"]])

# Pandas 05
df2 = df2.groupby("city")["grade"].mean()
print(f"Pandas Q5:")
print(df2)

# Pandas 06
df2 = df.copy()
df2 = df2.rename(columns={"Austin":"Houston"})
print(f"Pandas Q6:")
print(df2[["name","city"]])

# Pandas 07
df2 = df2.sort_values(by="grade",ascending=False)
print(f"Pandas Q7:\n {df2.head(3)}")

#_____________________________________________________________________________
# Numpy 01
np_1d = np.array([10, 20, 30, 40, 50])
print(f"Numpy Q1:")
print(f"Shape:{np_1d.shape}")
print(f"Dtype: {np_1d.dtype}")
print(f"Ndim: {np_1d.ndim}")

# Numpy 02
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"Numpy Q2:")
print(f"Shape:{arr.shape}")
print(f"Size: {arr.size}")

# Numpy 03
arr = arr[0:2,0:2] #start:end:step or sss (start:stop:step) --> [row:row,col:col]
print(f"Numpy Q3:\n {arr}")

# Numpy 04
arr1 = np.zeros((3,4))
arr2 = np.zeros((2,5))
print(f"Numpy Q4:")
print(f"3x4:\n {arr1} \n2x5:\n {arr2}")

# Numpy 05
aran = np.arange(0,50,5)
print(f"Numpy Q5:\n")
print(f"Array:\n {aran}")
print(f"Shape: {aran.shape}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
print(f"Standard Deviation: {arr.std()}")

# Numpy 06
ran_val = np.random.normal(0,1,200)
print(f"Numpy Q6:")
print(f"Mean: {ran_val.mean()}")
print(f"Standard Deviation: {ran_val.std()}")

#_____________________________________________________________________________

# Matplotlib 01
x = [0,1,2,3,4,5]
y = [0,1,4,9,16,25]
plt.plot(x,y,color="Blue",marker='o',linestyle='-')
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplotlib 02
subjects = ["Math","Science","English","History"]
scores = [88,92,75,83]
plt.bar(subjects,scores,color=["Red","Blue","Green","Yellow"])
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.show()

# Matplotlib 03
x1, y1 = [1,2,3,4,5],[2,4,5,4,5]
x2, y2 = [1,2,3,4,5],[5,4,3,2,1]

fig, ax = plt.subplots()
ax.scatter(x1,y1,color="orange",label="Dataset 1")
ax.scatter(x2,y2,color="black",label="Dataset 2")

ax.legend(loc="lower left")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.show()

# Matplotlib 04
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title("Squares")
ax1.plot(x,y,color="Purple")
ax2.set_title("Subject Scores")
ax2.bar(subjects,scores,color=["Blue","Red","Yellow","Green"])
plt.tight_layout()
plt.show()

#_____________________________________________________________________________
# Descriptive Stats 01
data = [12,15,14,10,18,22,13,16,14,15]
mean = np.mean(data)
median = np.median(data)
variance = np.var(data)
standard = np.std(data)

print(f"Descriptive Stats 01:\n")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {variance}")
print(f"Standard Deviation {standard}")

# Descriptive Stats 02
ran_val = np.random.normal(65,10,500)
plt.hist(ran_val,bins=20,color="turquoise")
plt.title("Distribution of Scores")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Descriptive Stats 03
group_a = [55,60,63,70,68,62,58,65]
group_b = [75,80,78,90,85,79,82,88]
plt.boxplot(x=[group_a,group_b], positions=[1,2], labels=["Group A","Group B"],patch_artist=True,medianprops={'color':'pink'})
plt.title("Score Comparison")
plt.show()

# Descriptive Stats 04
normal_data = np.random.normal(50,5,200)
skewed_data = np.random.exponential(10, 200)

# Creates subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plots Boxplot for Data 1
axs[0].set_title('Distribution Comparison')
axs[0].boxplot(normal_data,labels=["Normal"],patch_artist=True,medianprops={'color':'red'})

# Plots Boxplot for Data 2
axs[1].set_title('Distribution Comparison')
axs[1].boxplot(skewed_data,labels=["Exponential"],patch_artist=True,medianprops={'color':'blue'})
plt.show()

#The exponential distribution is more skewed. 
#The median would provide a more appropriate measure of central tendency for each distribution.

# Descriptive Stats 05
data1 = [10,12,12,16,18]
data1_df = pd.DataFrame(data1)

data2 = [10,12,12,16,150]
data2_df = pd.DataFrame(data2)

print("Descriptive Stats 05:\n")
print(f"Data 1 mean: {data1_df.mean()}")
print(f"Data 2 mean: {data2_df.mean()}")

print(f"Data 1 median: {data1_df.median()}")
print(f"Data 2 median: {data2_df.median()}")

print(f"Data 1 mode: {data1_df.mode()}")
print(f"Data 2 mode: {data2_df.mode()}")

# Data2's mean is different because of having an outlier number of 150.
# The median just eliminates the outer numbers in the array and takes the middle result.
# If there's multiple median numbers, the average of those two numbers is the median.
# Descriptive Stats 05
data1 = [10,12,12,16,18]
data1_df = pd.DataFrame(data1)

data2 = [10,12,12,16,150]
data2_df = pd.DataFrame(data2)

print("Descriptive Stats 05:\n")
print(f"Data 1 mean: {data1_df.mean()}")
print(f"Data 2 mean: {data2_df.mean()}")

print(f"Data 1 median: {data1_df.median()}")
print(f"Data 2 median: {data2_df.median()}")

print(f"Data 1 mode: {data1_df.mode()}")
print(f"Data 2 mode: {data2_df.mode()}")

# Data2's mean is different because of having an outlier number of 150.
# The median just eliminates the outer numbers in the array and takes the middle result.
# If there's multiple median numbers, the average of those two numbers is the median.

#_____________________________________________________________________________

# Hypothesis Question 01
group_a = [72,68,75,70,69,73,71,74]
group_b = [80,85,78,83,82,86,79,84]

res = ttest_ind(group_a, group_b)
print(f"Hypothesis Question 01:\n")
print(f"Statistic: {res.statistic}")
print(f"Pvalue: {res.pvalue}")

