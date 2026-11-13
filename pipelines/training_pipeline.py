# """Vertex AI pipeline: validate -> train -> compare with champion -> register if better.

# Each component is a thin wrapper: it moves data in and out, and calls tested
# functions from the eb_rampup package for the real work.
# """

# import os
# from typing import NamedTuple

# from kfp import dsl
# from kfp.dsl import Dataset, Input, Metrics, Model, Output

# # The image is fixed when components are defined, so it comes from an
# # environment variable. The placeholder lets tests compile without a real image.
# TRAINER_IMAGE = os.environ.get("TRAINER_IMAGE", "trainer:placeholder")
# SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


# @dsl.component(base_image=TRAINER_IMAGE)
# def validate_data(project: str, table: str, min_rows: int, data: Output[Dataset]):
#     from google.cloud import bigquery

#     from eb_rampup.bq import load_table
#     from eb_rampup.validation import validate_groups

#     df = load_table(bigquery.Client(project=project), table)
#     problems = validate_groups(df, min_rows=min_rows)
#     if problems:
#         raise ValueError("Data validation failed:\n- " + "\n- ".join(problems))

#     df.to_parquet(data.path, index=False)
#     data.metadata["row_count"] = len(df)


# @dsl.component(base_image=TRAINER_IMAGE)
# def train(data: Input[Dataset], model: Output[Model], metrics: Output[Metrics]):
#     import pandas as pd

#     from eb_rampup.features import TARGET, build_features, sample_weights
#     from eb_rampup.train import evaluate, save_artifacts, split, train_model

#     df = pd.read_parquet(data.path)
#     train_df, test_df = split(df)
#     fitted = train_model(build_features(train_df), train_df[TARGET], sample_weights(train_df))
#     results = evaluate(fitted, train_df, test_df)

#     save_artifacts(fitted, model.path, results)  # model.path is a folder under /gcs/
#     for name, value in results.items():
#         metrics.log_metric(name, value)  # shows up in the pipeline console


# @dsl.component(base_image=TRAINER_IMAGE)
# def compare_with_champion(
#     model: Input[Model], project: str, region: str, model_name: str, min_improvement: float
# ) -> NamedTuple("Outputs", [("promote", bool), ("reason", str)]):
#     import json
#     from collections import namedtuple
#     from pathlib import Path

#     from google.cloud import aiplatform

#     from eb_rampup.promotion import should_promote
#     from eb_rampup.train import to_local_path

#     aiplatform.init(project=project, location=region)
#     challenger = json.loads((Path(model.path) / "metrics.json").read_text())

#     champion = None
#     existing = aiplatform.Model.list(filter=f'display_name="{model_name}"')
#     if existing:
#         registry = aiplatform.models.ModelRegistry(existing[0])
#         champ_version = next(
#             (v for v in registry.list_versions() if "champion" in v.version_aliases),
#             None,
#         )
#         if champ_version:
#             champ = registry.get_model(version=champ_version.version_id)
#             champion = json.loads((to_local_path(champ.uri) / "metrics.json").read_text())
#             print(f"Champion is version {champ_version.version_id}")
#         else:
#             print("Model exists, but no version has the champion alias")

#     promote, reason = should_promote(challenger, champion, min_improvement)
#     print(reason)
#     return namedtuple("Outputs", ["promote", "reason"])(promote, reason)


# @dsl.component(base_image=TRAINER_IMAGE)
# def register_model(
#     model: Input[Model], project: str, region: str, model_name: str,
#     serving_image: str, git_sha: str,
# ):
#     import json
#     from pathlib import Path

#     from google.cloud import aiplatform

#     aiplatform.init(project=project, location=region)
#     metrics = json.loads((Path(model.path) / "metrics.json").read_text())
#     existing = aiplatform.Model.list(filter=f'display_name="{model_name}"')

#     registered = aiplatform.Model.upload(
#         display_name=model_name,
#         artifact_uri=model.uri,
#         serving_container_image_uri=serving_image,
#         parent_model=existing[0].resource_name if existing else None,
#         is_default_version=True,
#         version_aliases=["champion"],  # moves the alias off the old champion
#         version_description=f"MAE {metrics['mae']:.4f} | promoted by pipeline",
#         labels={"git_sha": git_sha},
#     )
#     print(f"Registered version {registered.version_id} as champion")


# @dsl.pipeline(name="loss-ratio-training")
# def training_pipeline(
#     project: str,
#     table: str = "eb_rampup.groups",
#     region: str = "us-central1",
#     model_name: str = "loss-ratio-ltd",
#     serving_image: str = SERVE_IMAGE,
#     min_rows: int = 1_000,
#     min_improvement: float = 0.0,
#     git_sha: str = "unknown",
# ):
#     validated = validate_data(project=project, table=table, min_rows=min_rows)
#     trained = train(data=validated.outputs["data"])
#     compared = compare_with_champion(
#         model=trained.outputs["model"], project=project, region=region,
#         model_name=model_name, min_improvement=min_improvement,
#     )
#     # `== True` is required: this builds a pipeline condition, not a Python if
#     with dsl.If(compared.outputs["promote"] == True, name="beats-champion"):
#         register_model(
#             model=trained.outputs["model"], project=project, region=region,
#             model_name=model_name, serving_image=serving_image, git_sha=git_sha,
#         )

"""Vertex AI pipeline: validate -> train -> compare with champion -> register.

Each component is a thin wrapper: it moves data in and out, and calls tested
functions from the eb_rampup package for the real work.

This change from the above code allows test data so the challenger and champion can both use the new data
Alias can now be "champion" on auto-promote or pending review if human needs to review
auto_promote can be set to false to always require human review
"""

import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# The image is fixed when components are defined, so it comes from an
# environment variable. The placeholder lets tests compile without a real image.
TRAINER_IMAGE = os.environ.get("TRAINER_IMAGE", "trainer:placeholder")
SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"


@dsl.component(base_image=TRAINER_IMAGE)
def validate_data(project: str, table: str, min_rows: int, data: Output[Dataset]):
    from google.cloud import bigquery

    from eb_rampup.bq import load_table
    from eb_rampup.validation import validate_groups

    df = load_table(bigquery.Client(project=project), table)
    problems = validate_groups(df, min_rows=min_rows)
    if problems:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(problems))

    df.to_parquet(data.path, index=False)
    data.metadata["row_count"] = len(df)


@dsl.component(base_image=TRAINER_IMAGE)
def train(
    data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    test_data: Output[Dataset],          # NEW: so the comparison can reuse the same rows
):
    import pandas as pd

    from eb_rampup.features import TARGET, build_features, sample_weights
    from eb_rampup.train import evaluate, save_artifacts, split, train_model

    df = pd.read_parquet(data.path)
    train_df, test_df = split(df)
    fitted = train_model(build_features(train_df), train_df[TARGET], sample_weights(train_df))
    results = evaluate(fitted, train_df, test_df)

    save_artifacts(fitted, model.path, results)  # model.path is a folder under /gcs/
    for name, value in results.items():
        metrics.log_metric(name, value)  # shows up in the pipeline console
    test_df.to_parquet(test_data.path, index=False)


@dsl.component(base_image=TRAINER_IMAGE)
def compare_with_champion(
    model: Input[Model], test_data: Input[Dataset], project: str, region: str,
    model_name: str, min_improvement: float,
) -> NamedTuple("Outputs", [("promote", bool), ("reason", str)]):
    """Score BOTH models on the challenger's test rows, so they're judged on the same data."""
    import json
    import tempfile
    from collections import namedtuple
    from pathlib import Path

    import pandas as pd

    from eb_rampup.features import TARGET, sample_weights
    from eb_rampup.promotion import should_promote
    from eb_rampup.registry import download_dir, find_champion
    from eb_rampup.serving import FeatureMismatchError, load_model_dir
    from eb_rampup.train import weighted_mae

    test_df = pd.read_parquet(test_data.path)
    y, w = test_df[TARGET], sample_weights(test_df)

    challenger = json.loads((Path(model.path) / "metrics.json").read_text())
    challenger["mae"] = weighted_mae(y, load_model_dir(model.path).predict(test_df), w)

    champion = None
    found = find_champion(project, region, model_name)
    if found is None:
        print("No champion yet")
    else:
        local = download_dir(found.uri, tempfile.mkdtemp(), project)
        try:
            champ_model = load_model_dir(local, version=str(found.version_id))
            champion = {"mae": weighted_mae(y, champ_model.predict(test_df), w)}
            print(f"Champion is version {found.version_id}; "
                  f"on this test data its MAE is {champion['mae']:.4f}")
        except FeatureMismatchError as e:
            print(f"Champion can't score today's features, so it can't be compared: {e}")

    promote, reason = should_promote(challenger, champion, min_improvement)
    print(reason)
    return namedtuple("Outputs", ["promote", "reason"])(promote, reason)


@dsl.component(base_image=TRAINER_IMAGE)
def register_model(
    model: Input[Model], project: str, region: str, model_name: str,
    serving_image: str, git_sha: str, alias: str,      # NEW: "champion" or "pending-review"
):
    import json
    from pathlib import Path

    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    metrics = json.loads((Path(model.path) / "metrics.json").read_text())
    existing = aiplatform.Model.list(filter=f'display_name="{model_name}"')

    registered = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_image,
        parent_model=existing[0].resource_name if existing else None,
        is_default_version=(alias == "champion"),
        version_aliases=[alias],  # an alias moves off whichever version had it
        version_description=f"MAE {metrics['mae']:.4f} | registered as {alias}",
        labels={"git_sha": git_sha},
    )

    # Verify instead of assuming: fail loudly if the alias didn't land
    registry = aiplatform.models.ModelRegistry(registered)
    holders = [v.version_id for v in registry.list_versions() if alias in v.version_aliases]
    if holders != [registered.version_id]:
        raise RuntimeError(f"Expected version {registered.version_id} to be {alias}, found {holders}")
    print(f"Registered version {registered.version_id} as {alias}")


@dsl.pipeline(name="loss-ratio-training")
def training_pipeline(
    project: str,
    table: str = "eb_rampup.groups",
    region: str = "us-central1",
    model_name: str = "loss-ratio-ltd",
    serving_image: str = SERVE_IMAGE,
    min_rows: int = 1_000,
    min_improvement: float = 0.0,
    git_sha: str = "unknown",
    auto_promote: bool = True,           # NEW: False sends a winner to human review
):
    validated = validate_data(project=project, table=table, min_rows=min_rows)
    trained = train(data=validated.outputs["data"])
    compared = compare_with_champion(
        model=trained.outputs["model"], test_data=trained.outputs["test_data"],
        project=project, region=region, model_name=model_name,
        min_improvement=min_improvement,
    )
    # `== True` is required: this builds a pipeline condition, not a Python if
    with dsl.If(compared.outputs["promote"] == True, name="beats-champion"):
        with dsl.If(auto_promote == True, name="auto-promote"):
            register_model(
                model=trained.outputs["model"], project=project, region=region,
                model_name=model_name, serving_image=serving_image, git_sha=git_sha,
                alias="champion",
            )
        with dsl.Else():
            register_model(
                model=trained.outputs["model"], project=project, region=region,
                model_name=model_name, serving_image=serving_image, git_sha=git_sha,
                alias="pending-review",
            )