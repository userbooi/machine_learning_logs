import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/music.csv")

# print(df)

# decide the features and targets
# in this data, the age and gender are the features (independent variable)
#               the genre is the target (dependent variable)

# split up the data
X = df.drop(columns=["genre"])
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(type(X_test))
# print(X)
# print(y)

# create the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test.values)

# print(pred)

print(accuracy_score(y_test, pred))
