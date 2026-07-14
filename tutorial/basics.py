import pandas as pd

df = pd.read_csv("data/vgsales.csv")

print(df)
print(df.values)
print(type(df.values))