from pathlib import Path

import click
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import KFold, cross_validate
from joblib import dump
import mlflow
import mlflow.sklearn

from .data import get_dataset
from .pipeline import create_pipeline_knn, create_pipeline_rfc

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)

@click.option(
    "-m",
    "--model",
    default='knn',
    type=str,
)

@click.option(
    "-sca",
    "--use-scaler",
    default=True,
    type=bool,
)

@click.option(
    "-thresh",
    "--use-threshold",
    default=False,
    type=bool
)

@click.option(
    "-pca",
    "--n-pca",
    default=None,
    type=int,
)

@click.option("--random-state", default=42, type=int)

@click.option(
    "--test-size",
    default=0.1,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)

@click.option("--kfold", default=5, type=int)

@click.option("--n-neighbors", default=5, type=int)

@click.option("--weights", default='uniform', type=str)

@click.option("--algorithm", default='auto', type=str)

@click.option("--n-estimators", default=100, type=int)

@click.option("--criterion", default='gini', type=str)

@click.option("--max-depth", default=None, type=int)

@click.option("--bootstrap", default=True, type=bool)

def train(dataset_path: Path, save_model_path: Path, test_size: float, random_state: int, use_scaler: bool,
          n_pca: int, use_threshold: bool, kfold: int, n_neighbors: int, weights: str, algorithm: str,
          model: str, n_estimators: int, criterion: str, max_depth: int, bootstrap: bool
          ) -> None:

    features_train, features_val, target_train, target_val =\
        get_dataset(dataset_path, test_size, random_state)

    with mlflow.start_run():
        mlflow.log_param("model", model)
        if model == 'rfc':
            pipeline = create_pipeline_rfc(use_scaler=use_scaler, use_threshold=use_threshold, n_estimators=n_estimators,
                n_pca=n_pca, criterion=criterion, max_depth=max_depth, bootstrap=bootstrap, random_state=random_state)\
                .fit(features_train, target_train)
            click.echo(f"Number features after selection: {pipeline['classifier'].n_features_in_}.")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("bootstrap", bootstrap)
        else:
            pipeline = create_pipeline_knn(use_threshold=use_threshold, use_scaler=use_scaler, n_neighbors=n_neighbors,
                                           weights=weights, algorithm=algorithm, n_pca=n_pca) \
                .fit(features_train, target_train)
            click.echo(f"Number features after selection: {pipeline['classifier'].n_features_in_}.")
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("weights", weights)
            mlflow.log_param("algorithm", algorithm)

        mlflow.log_param("use_scaler", use_scaler)
        scoring = ['accuracy', 'f1_macro', 'jaccard_macro']
        scores = cross_validate(pipeline, features_train, target_train, cv=KFold(kfold), scoring=scoring)
        mlflow.log_metric("accuracy", scores['test_accuracy'].mean())
        mlflow.log_metric("f1 score", scores['test_f1_macro'].mean())
        mlflow.log_metric("jaccard", scores['test_jaccard_macro'].mean())
        mlflow.sklearn.log_model(pipeline, 'model')
        click.echo(f"Scores: {scores}\n"
                   f"Accuracy: {scores['test_accuracy'].mean()}, "
                   f"f1 score: {scores['test_f1_macro'].mean()}, "
                   f"jaccard score {scores['test_jaccard_macro'].mean()}")
        dump(scores, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")