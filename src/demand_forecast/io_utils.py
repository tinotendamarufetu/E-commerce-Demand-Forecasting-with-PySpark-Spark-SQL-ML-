# src/demand_forecast/io_utils.py

import os
from pyspark.ml.pipeline import PipelineModel


def save_pipeline_model(model: PipelineModel, save_dir: str, experiment_name: str) -> str:
    """
    Save a Spark PipelineModel to a directory:
    models/<experiment_name>/

    Returns the full path.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, experiment_name)

    print(f"Saving model to: {path}")
    # Overwrite if it already exists
    model.write().overwrite().save(path)
    return path
