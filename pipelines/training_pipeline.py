from kfp import dsl, component, compiler
import eb_rampup
from eb_rampup.train import load_data
from eb_rampup.features import build_features, sample_weights, INDUSTRY_RISK,REGIONS,ELIMINATION_DAYS,BENEFIT_PCTS,YEARS,BASE_LOSS_RATIO,PREMIUM_RATE


@component(packages_to_install=["google-cloud-bigquery"])
def validation(project, table, region):

    data = load_data(['--project', project, '--table' ,table])
    features = build_features(data)

    # runtime validation
    

@component()
def training(
    project, 
    table, 
    row_count, 
    output_uri
):
    eb_rampup.train.main(argv=["--project", project, "--table", table, "--n-rows", row_count, "--model-dir", output_uri])

@component()
def compare_and_register():
    pass

@dsl.pipeline(name="training-pipeline")
def loss_ratio_pipeline():

    validation = validate_data(project=project, table=table)

    training = train_and_evaluate(
        project=project,
        table=table,
        row_count=validation.output,
        package_uri=package_uri, 
        output_uri=output_uri
    )

    compare_and_register(
        metrics_uri=training.outputs["metrics_uri"],
        model_uri=training.outputs["model_uri"],
        project=project
    )