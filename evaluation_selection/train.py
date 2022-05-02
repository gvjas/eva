from pathlib import Path

import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from joblib import dump
import mlflow
import mlflow.sklearn

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

def train(dataset_path: Path, save_model_path: Path, random_state: int, test_size: float,
          kfold: int, n_neighbors: int, weights: str, algorithm: str) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run():
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                          n_jobs=-1).fit(features_train, target_train)
        target_pred = classifier.predict(features_val)
        scoring = ['accuracy', 'f1_macro', 'jaccard_macro']
        scores = cross_validate(classifier, features_train, target_train, cv=KFold(kfold), scoring=scoring)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("weights", weights)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_metric("accuracy", scores['test_accuracy'].mean())
        mlflow.log_metric("f1 score", scores['test_f1_macro'].mean())
        mlflow.log_metric("jaccard", scores['test_jaccard_macro'].mean())
        mlflow.sklearn.log_model(classifier, 'model')
        click.echo(f"Scores: {scores}\n"
                   f"Accuracy: {scores['test_accuracy'].mean()}, "
                   f"f1 score: {scores['test_f1_macro'].mean()}, "
                   f"jaccard score {scores['test_jaccard_macro'].mean()}")
        dump(scores, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")