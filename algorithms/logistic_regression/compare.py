import time
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression as SklearnLR
from model import LogisticRegression as MyLR

'''
compares my LogisticRegression (model.py) with scikit-learn's LogisticRegression
on the breast cancer dataset, with raw features and with standardized features

note: sklearn uses L2 regularization by default (C=1.0), so "sklearn (no penalty)"
is the closest match to my model, which has no regularization
'''

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features (mean 0, std 1) using only the training data statistics
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def evaluate(name, model, X_tr, X_te):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_tr, y_train)
    train_time = time.perf_counter() - start

    y_preds = model.predict(X_te)

    # probabilities for ROC AUC
    if isinstance(model, MyLR):
        y_probs = model.sigmoid_function(np.dot(X_te, model.w) + model.b)
    else:
        y_probs = model.predict_proba(X_te)[:, 1]

    return {
        "name": name,
        "acc": accuracy_score(y_test, y_preds),
        "prec": precision_score(y_test, y_preds),
        "rec": recall_score(y_test, y_preds),
        "f1": f1_score(y_test, y_preds),
        "auc": roc_auc_score(y_test, y_probs),
        "time": train_time,
        "warnings": len(caught),
        "model": model,
    }


def print_table(title, results):
    print(f"\n=== {title} ===")
    print(f"{'model':<28}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'auc':>7}{'time(s)':>9}{'warns':>7}")
    for r in results:
        print(f"{r['name']:<28}{r['acc']:>7.3f}{r['prec']:>7.3f}{r['rec']:>7.3f}"
              f"{r['f1']:>7.3f}{r['auc']:>7.3f}{r['time']:>9.4f}{r['warnings']:>7}")


def make_models():
    return [
        ("mine (lr=0.001, 1000 ep)", MyLR()),
        ("mine (lr=0.1, 5000 ep)", MyLR(lr=0.1, epochs=5000)),
        ("sklearn (default, L2)", SklearnLR(max_iter=10000)),
        ("sklearn (no penalty)", SklearnLR(penalty=None, max_iter=10000)),
    ]


raw_results = [evaluate(name, m, X_train, X_test) for name, m in make_models()]
print_table("raw features", raw_results)

scaled_results = [evaluate(name, m, X_train_s, X_test_s) for name, m in make_models()]
print_table("standardized features", scaled_results)

# how similar are the learned weights? (standardized features, default sklearn)
mine = scaled_results[1]["model"]
sk = scaled_results[2]["model"]
corr = np.corrcoef(mine.w, sk.coef_[0])[0, 1]
agree = np.mean(mine.predict(X_test_s) == sk.predict(X_test_s))
print("\n=== mine (lr=0.1) vs sklearn default, standardized ===")
print(f"weight correlation:   {corr:.3f}")
print(f"prediction agreement: {agree*100:.1f}%")
print(f"bias: mine={mine.b:.3f}  sklearn={sk.intercept_[0]:.3f}")
