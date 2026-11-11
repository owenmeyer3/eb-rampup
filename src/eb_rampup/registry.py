"""Find and download the champion model from the Vertex AI Model Registry."""

from pathlib import Path

from eb_rampup.serving import LoadedModel, load_model_dir


def find_champion(project: str, region: str, model_name: str):
    """Return the registry version aliased 'champion', or None."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    models = aiplatform.Model.list(filter=f'display_name="{model_name}"')
    if not models:
        return None
    registry = aiplatform.models.ModelRegistry(models[0])
    for version in registry.list_versions():
        if "champion" in version.version_aliases:
            return registry.get_model(version=version.version_id)
    return None


def download_dir(gcs_uri: str, dest, project:str) -> Path:
    """Copy every file under a gs:// folder to a local folder."""
    from google.cloud import storage

    bucket, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    dest = Path(dest)
    for blob in storage.Client(project=project).list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        target = dest / blob.name[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
    return dest


def load_champion(project: str, region: str, model_name: str, dest="/tmp/champion") -> LoadedModel:
    champion = find_champion(project, region, model_name)
    if champion is None:
        raise RuntimeError(f"No version of {model_name} has the champion alias")
    local = download_dir(champion.uri, dest, project)
    return load_model_dir(local, version=str(champion.version_id))
