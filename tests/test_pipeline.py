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

def registration_aliases(spec) -> set[str]:
    """Every alias passed to a task, wherever it sits in the pipeline."""
    aliases = set()
    for component in spec["components"].values():
        for task in component.get("dag", {}).get("tasks", {}).values():
            alias = task.get("inputs", {}).get("parameters", {}).get("alias", {})
            if "runtimeValue" in alias:
                aliases.add(alias["runtimeValue"]["constant"])
    return aliases

def inspect_pipeline(path='tests/inspect_pipeline.yaml'):
    # manual call to view pipeline in testing
    # python -c "from tests.test_pipeline import inspect_pipeline; inspect_pipeline()"
    from kfp import compiler
    from pipelines.training_pipeline import training_pipeline
    compiler.Compiler().compile(training_pipeline, path)

def test_pipeline_compiles(spec):
    assert spec["pipelineInfo"]["name"] == "loss-ratio-training"
    tasks = spec["root"]["dag"]["tasks"]
    assert 'validate-data' in tasks
    assert 'train' in tasks
    assert 'compare-with-champion' in tasks
    assert 'register-model' not in tasks
    assert 'auto_promote' in spec["root"]["inputDefinitions"]["parameters"]
    assert registration_aliases(spec) == {"champion", "pending-review"}

    reg_exists = False
    components = spec['components']
    for k in components:
        if "dag" in components[k] and "tasks" in components[k]["dag"] and "register-model" in components[k]["dag"]["tasks"]:
            reg_exists = True

    assert reg_exists

