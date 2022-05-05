from pathlib import Path
from typing import Tuple, Any
import click

import numpy as np
from joblib import dump
import mlflow
import mlflow.sklearn

from .data import get_dataset
from .pipeline import create_pipeline_knn, create_pipeline_rfc
from .nested import nested_cv


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
    default="knn",
    type=str,
)
@click.option(
    "-sca",
    "--use-scaler",
    default=False,
    type=bool,
)
@click.option("-thresh", "--use-threshold", default=True, type=bool)
@click.option(
    "-pca",
    "--n-pca",
    default=None,
    type=int,
)
@click.option(
    "-kbest",
    "--use-kbest",
    default=False,
    type=bool,
)
@click.option(
    "-sfm",
    "--use-sfm",
    default=False,
    type=bool,
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-size",
    default=0.00001,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
@click.option("--kfold", default=5, type=int)
@click.option("--knested", default=5, type=int)
@click.option("--n-neighbors", default=5, type=int)
@click.option("--weights", default="distance", type=str)
@click.option("--algorithm", default="auto", type=str)
@click.option("--n-estimators", default=100, type=int)
@click.option("--max-features", default="auto", type=str)
@click.option("--criterion", default="gini", type=str)
@click.option("--max-depth", default=None, type=int)
@click.option("--bootstrap", default=False, type=bool)
def train(
    dataset_path: Path,
    save_model_path: Path,
    test_size: float,
    random_state: int,
    use_scaler: bool,
    use_threshold: bool,
    use_kbest: bool,
    n_pca: int,
    use_sfm: bool,
    kfold: int,
    knested: int,
    n_neighbors: int,
    weights: str,
    algorithm: str,
    model: str,
    n_estimators: int,
    max_features: Any,
    criterion: str,
    max_depth: int,
    bootstrap: bool,
) -> None:
    split_dataset = get_dataset(dataset_path, test_size, random_state)
    features_train, features_val, target_train, target_val = [
        df.to_numpy() for df in split_dataset
    ]

    if model == "rfc":
        pipeline = create_pipeline_rfc(
            use_scaler=use_scaler,
            use_threshold=use_threshold,
            use_kbest=use_kbest,
            n_pca=n_pca,
            use_sfm=use_sfm,
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            max_depth=max_depth,
            bootstrap=bootstrap,
            random_state=random_state,
        )
        rfc = pipeline.steps.pop()
        if any([use_threshold, use_scaler, use_kbest, n_pca, use_sfm]):
            selector = pipeline.fit_transform(features_train, target_train)
            click.echo(f"Number features after selection: {selector.shape}.")
        else:
            selector = features_train
        space = dict()
        space["n_estimators"] = range(10, 300, 10)
        # space['max_features'] = ['auto', 'sqrt', 'log2']
        # space['criterion'] = ['gini', 'entropy']
        space["max_depth"] = range(1, 50)
        # space['bootstrap'] = [True, False]
        scores = nested_cv(
            rfc[1], selector, target_train, space, kfold, knested, random_state
        )
        pipeline.steps.append(rfc)

        for i in range(knested):
            with mlflow.start_run():
                mlflow.log_param("model", model.upper() + "_nested_cv")
                mlflow.log_param(
                    "n_estimators",
                    scores["best_params"][i].get("n_estimators", n_estimators),
                )
                mlflow.log_param(
                    "max_features",
                    scores["best_params"][i].get("max_features", max_features),
                )
                mlflow.log_param(
                    "criterion", scores["best_params"][i].get("criterion", criterion)
                )
                mlflow.log_param(
                    "max_depth", scores["best_params"][i].get("max_depth", max_depth)
                )
                mlflow.log_param(
                    "bootstrap", scores["best_params"][i].get("bootstrap", bootstrap)
                )
                mlflow.log_param("use_threshold", use_threshold)
                mlflow.log_param("use_scaler", use_scaler)
                mlflow.log_param("n_pca", n_pca)
                mlflow.log_param("use_sfm", use_sfm)
                mlflow.log_param("use_kbest", use_kbest)
                acc = round(scores["test_accuracy"][i], 3)
                f1 = round(scores["test_f1_macro"][i], 3)
                jac = round(scores["test_jaccard_macro"][i], 3)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1 score", f1)
                mlflow.log_metric("jaccard", jac)
                dump(pipeline, save_model_path)
    else:
        pipeline = create_pipeline_knn(
            use_threshold=use_threshold,
            use_scaler=use_scaler,
            use_kbest=use_kbest,
            n_pca=n_pca,
            use_sfm=use_sfm,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            random_state=random_state,
        ).fit(features_train, target_train)
        rfc = pipeline.steps.pop()
        if any([use_threshold, use_scaler, use_kbest, n_pca, use_sfm]):
            selector = pipeline.fit_transform(features_train, target_train)
            click.echo(f"Number features after selection: {selector.shape}.")
        else:
            selector = features_train
        space = dict()
        space["n_neighbors"] = range(1, 10)
        # space['weights'] = ['uniform', 'distance']
        space["algorithm"] = ["auto", "ball_tree", "kd_tree", "brute"]
        scores = nested_cv(
            rfc[1], selector, target_train, space, kfold, knested, random_state
        )
        pipeline.steps.append(rfc)

        for i in range(knested):
            with mlflow.start_run():
                mlflow.log_param("model", model.upper() + "_nested_cv")
                mlflow.log_param(
                    "n_neighbors",
                    scores["best_params"][i].get("n_neighbors", n_neighbors),
                )
                mlflow.log_param(
                    "weights", scores["best_params"][i].get("weights", weights)
                )
                mlflow.log_param(
                    "algorithm", scores["best_params"][i].get("algorithm", algorithm)
                )
                mlflow.log_param("use_threshold", use_threshold)
                mlflow.log_param("use_scaler", use_scaler)
                mlflow.log_param("n_pca", n_pca)
                mlflow.log_param("use_sfm", use_sfm)
                mlflow.log_param("use_kbest", use_kbest)
                acc = round(scores["test_accuracy"][i], 3)
                f1 = round(scores["test_f1_macro"][i], 3)
                jac = round(scores["test_jaccard_macro"][i], 3)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1 score", f1)
                mlflow.log_metric("jaccard", jac)
                dump(pipeline, save_model_path)

    with mlflow.start_run():
        acc = round(np.mean(scores["test_accuracy"]), 3)
        f1 = round(np.mean(scores["test_f1_macro"]), 3)
        jac = round(np.mean(scores["test_jaccard_macro"]), 3)
        acc_std = round(np.std(scores["test_accuracy"]), 3)
        f1_std = round(np.std(scores["test_f1_macro"]), 3)
        jac_std = round(np.std(scores["test_jaccard_macro"]), 3)
        mlflow.log_param("model", model.upper() + "_avg_nested_cv")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("jaccard", jac)
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(
            f"Accuracy: {acc} ({acc_std}),\n"
            f"f1 score: {f1} ({f1_std}),\n"
            f"jaccard score {jac} ({jac_std})"
        )
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
