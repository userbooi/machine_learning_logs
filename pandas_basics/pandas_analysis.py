import numpy as np
import pandas as pd

'''
powerful data manipulation and analysis tool

works well with structures from numpy and matplot lib

pandas works with two core structures:
    - series: 1D
    - dataframe: 2D
'''

#=========================creating dataframes========================
# can be created from lists, dictionaries, and others
two_dimensional_list = [[1, 2, 3], [3, 4, 5]]
label_table = {
    "Name": ["player1", "player2", "player3"],
    "strength": [10, 23, 15],
    "Age": [21, 20, 23],
}

dataframe_list = pd.DataFrame(two_dimensional_list)
dataframe_table = pd.DataFrame(label_table)
dataframe_specific_col = pd.DataFrame(label_table, columns=["Name", "Age"])

print(dataframe_table)
print(dataframe_list)
print(dataframe_specific_col)

#=========================creating series============================
# can be created from lists, dictionaries, numpy arrays, and others
# age_series = pd.Series([25, 30, 35]) # default index
# print(age_series)
#
# age_series_with_labels = pd.Series(np.array([25, 30, 35]), index=["Player_A", "Player_B", "Player_C"]) # custom labels
# print(age_series_with_labels)

#=====================analyzing dataframes===========================
# data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
#         'Age': [24, 27, 22, 32, 29],
#         'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
# df = pd.DataFrame(data)
#
# print("First 3 rows using head():")
# print(df.head(3))
#
# print("\nLast 2 rows using tail():")
# print(df.tail(2))
#
# print("\nDataFrame summary using info():")
# df.info()
#
# # generate stats for the numerical columns
# print("\nDataFrame summary using describe()")
# print(df.describe())

#======================more on describe==========================
# describe is different for strings (Object type in pandas) than for numerical values

# data = pd.read_csv("data/nba.csv") # reads it and turns it into a dataframe
# print(data)
#
# print("\nSummary using describe:")
# print(data.describe())
#
# # customize what to include in the description
# include = ["object", "float", "int"]
#
# print("\nSummary using customized describe:")
# print(data.describe(include=include))
#
# # describing a specific one
# specific_data = data["Name"]
# print("\nSummary using describe on a specific column:")
# print(specific_data.describe())

#=====================indexing with pandas==========================
# # column(s)
# print("\nthe following are specific columns printed:")
# print(data["Team"])
# print("\n")
# print(data[["Name", "Team"]])
#
# # filters
# print("\nthe following are the players that are older than 35 years old:")
# print(data[data["Age"]>35])
#
# # using loc
# print("\n")
# #               row
# print(data.loc[109])
# print("\n")
# #                   rows
# print(data.loc[[109, 101, 93]])
# print("\n")
# #                   row         column
# print(data.loc[[109, 101], ["Name", "Team", "Position"]])
