from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline_knn(
    use_scaler: bool, n_neighbors: int, weights: str, algorithm: str
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                 n_jobs=-1)
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_rfc(
    use_scaler: bool, n_estimators: int, criterion: str, max_depth: int, bootstrap: bool
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                   max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        )
    )
    return Pipeline(steps=pipeline_steps)