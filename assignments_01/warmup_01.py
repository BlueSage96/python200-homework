# Pandas 01
import pandas as pd
data = {
    "name": ["Alice","Bob","Carol","David","Eve"],
    "grade": [85,72,90,68,95],
    "city": ["Boston","Austin","Boston","Denver","Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)
print(f"First three rows: {df.head(3)}")
print(f"Shape:{df.shape}")
print(f"Column data types: {df.info()}")

# Pandas 02
df1 = df[(df['grade'] > 80)]
print(f"Pandas Q2:\n {df1}")