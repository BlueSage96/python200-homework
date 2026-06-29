# Pandas 01
import pandas as pd
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
print(f"Column data types: {df.info()}")

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