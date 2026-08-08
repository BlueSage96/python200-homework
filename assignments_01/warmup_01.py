import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, ttest_rel, kstest, pearsonr
from scipy import stats
import seaborn as sns

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
df1 = df.copy()
df1 = df1[(df1['grade'] > 80) & (df1['passed']==True)]
print(f"\nPandas Q2:\n {df1}")

# Pandas 03
df2 = df.copy()
df2["grade_curved"] = df["grade"] + 5
print(f"\nPandas Q3:\n {df2}")

# Pandas 04
df2["name_upper"] = df2["name"].str.upper()
print(f"\nPandas Q4:")
print(df2[["name","name_upper"]])

# Pandas 05
df2 = df2.groupby("city")["grade"].mean()
print(f"\nPandas Q5:")
print(df2)

# Pandas 06
df2 = df.copy()
df2["city"] = df2["city"].replace({"Austin":"Houston"})
print(f"\nPandas Q6:")
print(df2[["name","city"]])

# Pandas 07
df = df.sort_values(by="grade",ascending=False)
print(f"\nPandas Q7:\n {df.head(3)}")

#_____________________________________________________________________________
# Numpy 01
np_1d = np.array([10, 20, 30, 40, 50])
print(f"\nNumpy Q1:")
print(f"Shape:{np_1d.shape}")
print(f"Dtype: {np_1d.dtype}")
print(f"Ndim: {np_1d.ndim}")

# Numpy 02
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"\nNumpy Q2:")
print(f"Shape:{arr.shape}")
print(f"Size: {arr.size}")

# Numpy 03
arr = arr[0:2,0:2] #start:end:step or sss (start:stop:step) --> [row:row,col:col]
print(f"\nNumpy Q3:\n {arr}")

# Numpy 04
arr1 = np.zeros((3,4))
arr2 = np.zeros((2,5))
print(f"\nNumpy Q4:")
print(f"3x4:\n {arr1} \n2x5:\n {arr2}")

# Numpy 05
aran = np.arange(0,50,5)
print(f"\nNumpy Q5:\n")
print(f"Array:\n {aran}")
print(f"Shape: {aran.shape}")
print(f"Mean: {aran.mean()}")
print(f"Sum: {aran.sum()}")
print(f"Standard Deviation: {aran.std()}")

# Numpy 06
ran_val = np.random.normal(0,1,200)
print(f"\nNumpy Q6:")
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

print(f"\nDescriptive Stats 01:\n")
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

'''
The exponential distribution is more skewed. 

The median would provide a more appropriate measure 
of central tendency for each distribution.
'''

# Descriptive Stats 05
data1 = [10,12,12,16,18]
data1_df = pd.DataFrame(data1)

data2 = [10,12,12,16,150]
data2_df = pd.DataFrame(data2)

print("\nDescriptive Stats 05:\n")
print(f"Data 1 mean: {data1_df.mean()}")
print(f"Data 2 mean: {data2_df.mean()}")

print(f"Data 1 median: {data1_df.median()}")
print(f"Data 2 median: {data2_df.median()}")

print(f"Data 1 mode: {data1_df.mode()}")
print(f"Data 2 mode: {data2_df.mode()}")

'''
Data2 contains an outlier (150), which increases the mean.
The median stays the same as data1 because it is not 
affected much by extreme values.
'''
#_____________________________________________________________________________

# Hypothesis Question 01
group_a = [72,68,75,70,69,73,71,74]
group_b = [80,85,78,83,82,86,79,84]

res = ttest_ind(group_a, group_b)
print(f"\nHypothesis Question 01:\n")
print(f"Statistic: {res.statistic}")
print(f"Pvalue: {res.pvalue}")

# Hypothesis Question 02
alpha = 0.05
res_pvalue = res.pvalue
print(f"\nHypothesis Question 02:\n")

if (res_pvalue >= alpha):
    print(f"Statistically significant based on p-value: {alpha}")
else:
    print(f"Not statistically significant based on p-value: {alpha}")
    
# Hypothesis Question 03
before = [60,65,70,58,62,67,63,66]
after = [68,70,76,65,69,72,70,71]

before_after = ttest_rel(before,after)
print(f"\nHypothesis Question 03:\n")
print(f"Statistic: {before_after.statistic}")
print(f"Pvalue: {before_after.pvalue}")

# Hypothesis Question 04
scores = [72,68,75,70,69,74,71,73]
new_scores = stats.ttest_1samp(scores,popmean=70)
print(f"\nHypothesis Question 04:\n")
print(f"Statistic: {new_scores.statistic}")
print(f"Pvalue: {new_scores.pvalue}")

# Hypothesis Question 05
res2 = ttest_ind(group_a, stats.norm.cdf, alternative='greater')
print(f"\nHypothesis Question 05:\n")
print(f"Pvalue: {res2.pvalue}")

# Hypothesis Question 06
print(f"\nHypothesis Question 06:\n")
print(
    "Group B had a higher average score than Group A.\n"
    "Because the p-value is much smaller than 0.05, the difference is unlikely to be due to chance."
)

#_____________________________________________________________________________

# Correlation Question 01
x = [1,2,3,4,5]
y = [2,4,6,8,10]
pearson = np.corrcoef(x,y)

print(f"\nCorrelation Question 01:\n")
print(f"Correlation matrix: {pearson}")
print(f"Correlation coefficient: {pearson[0,1]}")

# I expect the correlation to be 1 because y increases proportionally
# with x, creating a perfect positive linear relationship.

# Correlation Question 02
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

pearson2 = pearsonr(x,y)
print(f"\nCorrelation Question 02:\n")
print(f"Statistic: {pearson2.statistic}")
print(f"P-value: {pearson2.pvalue}")

# Corrlation Question 03
people = {
    "height": [160,165,170,175,180],
    "weight": [55,60,65,72,80],
    "age": [25,30,22,35,28]
}

df = pd.DataFrame(people)
people_corr = df.corr()
print(f"\nCorrelation Question 03:\n")
print(people_corr)

# Corrleation Question 04
x = [10,20,30,40,50]
y = [90,75,60,45,30]

fig, ax = plt.subplots()
ax.set_title("Negative Correlation")
ax.scatter(x,y,color="teal")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Corrleation Question 05
sns.heatmap(people_corr,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#_____________________________________________________________________________

# Pipeline Question 01
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr,name="values")

def clean_data(series):
    return series.dropna()
    
def summarize_data(series):
    series = ({
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    })
    return series

def data_pipeline(arr):
    created = create_series(arr)
    return summarize_data(clean_data(created))

result = data_pipeline(arr)

print(f"\n Pipeline Question 01: \n")

for key, value in result.items():
    print(f"{key}: {value}")