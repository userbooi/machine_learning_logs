import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import LogisticRegression

data = load_breast_cancer() # or set the parameter return_X_y=True
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_preds = model.predict(X_test)

acc = accuracy_score(y_test, y_preds)
print(f"The accuracy of the model is: {acc}")
