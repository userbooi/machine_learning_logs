import pandas as pd

dataframe = pd.read_csv("data/people_data.csv")
print(dataframe)

print("\n")

# specific columns
specific_dataframe = pd.read_csv("data/people_data.csv", usecols=["First Name", "Email"])
print(specific_dataframe)

print("\n")

# instead of numerical indexes, one of the columns can be used as the identifier
dictionary_dataframe = pd.read_csv("data/people_data.csv", index_col="Last Name")
print(dictionary_dataframe)
print("\n")
print(dictionary_dataframe["First Name"])
print("\n")
print(dictionary_dataframe.loc[["Terrell", "Travis"], ["Sex", "Job Title"]])
