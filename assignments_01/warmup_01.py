import pandas as pd
import numpy as np

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
print(f"shape:{np_1d.shape}")
print(f"dtype: {np_1d.dtype}")
print(f"ndim: {np_1d.ndim}")

# Numpy 02
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"Numpy Q2:")
print(f"shape:{arr.shape}")
print(f"size: {arr.size}")

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
print(f"array:\n {aran}")
print(f"shape: {aran.shape}")
print(f"mean: {arr.mean()}")
print(f"sum: {arr.sum()}")
print(f"standard deviation: {arr.std()}")

# Numpy 06
ran_val = np.random.normal(0,1,200)
print(f"Numpy Q6:")
print(f"mean: {ran_val.mean()}")
print(f"standard deviation: {ran_val.std()}")