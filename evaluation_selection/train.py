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
def train(dataset_path: Path) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=0.1, random_state=42
    )

    classifier = KNeighborsClassifier(n_jobs=-1).fit(features_train, target_train)

    target_pred = classifier.predict(features_val)
    scores = cross_validate(classifier, features_train, target_train, cv=KFold(3))
    click.echo(f"Scores: {scores}\n")