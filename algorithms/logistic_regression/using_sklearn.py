from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# bc_dataset = load_breast_cancer()
# print(bc_dataset.feature_names)
# print(bc_dataset.data.shape)

X, y = load_breast_cancer(return_X_y=True)
# # print(X)
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LogisticRegression(max_iter=10000, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc*100}%")
