import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from typing import Dict, List, Any


def nested_cv(
    model: object,
    X: pd.array,
    y: pd.array,
    space: Dict[str, Any],
    kfold: int,
    knested: int,
    random_state: int,
) -> Dict[str, List[Any]]:
    cv_outer = KFold(n_splits=knested, shuffle=True, random_state=random_state)
    outer_results: dict[str, list[Any]] = dict()
    outer_results["test_accuracy"] = []
    outer_results["test_f1_macro"] = []
    outer_results["test_jaccard_macro"] = []
    outer_results["best_params"] = []
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        cv_inner = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            model, space, scoring="accuracy", cv=cv_inner, refit=True, n_jobs=-1
        )
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average="macro")
        jac = jaccard_score(y_test, yhat, average="macro")
        outer_results["test_accuracy"].append(acc)
        outer_results["test_f1_macro"].append(f1)
        outer_results["test_jaccard_macro"].append(jac)
        outer_results["best_params"].append(result.best_params_)

        print(
            ">acc=%.3f, est=%.3f, f1=%.3f, jac=%.3f, cfg=%s"
            % (acc, result.best_score_, f1, jac, result.best_params_)
        )
    return outer_results
