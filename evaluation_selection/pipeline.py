from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
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