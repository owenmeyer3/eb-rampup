import pytest

pytest.importorskip("kfp")
yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    from kfp import compiler

    from pipelines.training_pipeline import training_pipeline

    out = tmp_path_factory.mktemp("pipeline") / "pipeline.yaml"
    compiler.Compiler().compile(training_pipeline, str(out))
    return yaml.safe_load(out.read_text())


def test_pipeline_compiles(spec):
    assert spec["pipelineInfo"]["name"] == "loss-ratio-training"
    tasks = spec["root"]["dag"]["tasks"]
    assert 'validate-data' in tasks
    assert 'train' in tasks
    assert 'compare-with-champion' in tasks
    assert 'register-model' not in tasks

    reg_exists = False
    components = spec['components']
    for k in components:
        if "dag" in components[k] and "tasks" in components[k]["dag"] and "register-model" in components[k]["dag"]["tasks"]:
            reg_exists = True

    assert reg_exists

