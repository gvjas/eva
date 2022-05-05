from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)
from typing import Union


def pipe_selectors(
    use_threshold: bool,
    use_scaler: bool,
    use_kbest: bool,
    use_sfm: bool,
    n_pca: int,
    random_state: int,
):
    pipeline_steps = []
    if use_threshold:
        pipeline_steps.append(("threshold", VarianceThreshold(0.01)))
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if n_pca:
        pipeline_steps.append(("pca", PCA(n_components=n_pca)))
    if use_kbest:
        pipeline_steps.append(("kbest", SelectKBest(mutual_info_classif)))
    if use_sfm:
        selection_model = ExtraTreesClassifier(random_state=random_state)
        pipeline_steps.append(("sfm", SelectFromModel(selection_model)))
    return pipeline_steps


def create_pipeline_knn(
    use_threshold: bool,
    use_scaler: bool,
    use_kbest: bool,
    use_sfm: bool,
    n_neighbors: int,
    weights: str,
    algorithm: str,
    n_pca: int,
    random_state: int,
) -> Pipeline:
    pipeline_steps = pipe_selectors(
        use_threshold, use_scaler, use_kbest, use_sfm, n_pca, random_state
    )

    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=-1
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_rfc(
    use_scaler: bool,
    use_threshold: bool,
    use_kbest: bool,
    n_pca: int,
    use_sfm: bool,
    n_estimators: int,
    max_features: Union[float, str],
    criterion: str,
    max_depth: int,
    bootstrap: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = pipe_selectors(
        use_threshold, use_scaler, use_kbest, use_sfm, n_pca, random_state
    )

    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                criterion=criterion,
                max_depth=max_depth,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    )

    return Pipeline(steps=pipeline_steps)
