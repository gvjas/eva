from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder,MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,  f_classif, mutual_info_classif

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV, SequentialFeatureSelector

def create_pipeline_knn(
    use_threshold: bool, use_scaler: bool, n_neighbors: int, weights: str, algorithm: str,
    n_pca: int
) -> Pipeline:
    pipeline_steps = []
    if use_threshold:
        pipeline_steps.append(("threshold", VarianceThreshold(0.01)))
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if n_pca:
        pipeline_steps.append(("pca", PCA(n_components=n_pca)))

    pipeline_steps.append(
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                 n_jobs=-1)
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_rfc(
    use_scaler: bool, use_threshold: bool, n_estimators: int, n_pca: int,
        criterion: str, max_depth: int, bootstrap: bool, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_threshold:
        pipeline_steps.append(("threshold", VarianceThreshold(0.01)))
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if n_pca:
        pipeline_steps.append(("pca", PCA(n_components=n_pca)))

    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                   bootstrap=bootstrap, random_state=random_state, n_jobs=-1)
        )
    )

    return Pipeline(steps=pipeline_steps)