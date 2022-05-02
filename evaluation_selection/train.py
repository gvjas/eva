from pathlib import Path

import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--random-state", default=42, type=int)

@click.option(
    "--test-size",
    default=0.1,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)

def train(dataset_path: Path, random_state: int, test_size: float) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    classifier = KNeighborsClassifier(n_jobs=-1).fit(features_train, target_train)

    target_pred = classifier.predict(features_val)
    scoring = ['accuracy', 'f1_macro', 'jaccard_macro']
    scores = cross_validate(classifier, features_train, target_train, cv=KFold(3), scoring=scoring)
    click.echo(f"Scores: {scores}\n"
               f"Accuracy: {scores['test_accuracy'].mean()}, "
               f"f1 score: {scores['test_f1_macro'].mean()}, "
               f"jaccard score {scores['test_jaccard_macro'].mean()}")