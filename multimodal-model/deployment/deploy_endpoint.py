"""
Deploy a PyTorch model to AWS SageMaker.

This script:
- Loads environment variables from `.env`
- Creates and deploys a SageMaker PyTorchModel endpoint

Required env vars:
- SAGEMAKER_ROLE
- SAGEMAKER_MODEL_URI
- SAGEMAKER_ENDPOINT_NAME

Run:
    python deploy_endpoint.py
"""

from sagemaker.pytorch import PyTorchModel
import sagemaker
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)


def deploy_endpoint():
    sagemaker.Session()
    role = os.environ.get("SAGEMAKER_ROLE")

    model_uri = os.environ.get("SAGEMAKER_MODEL_URI")

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",  # Use ml.g5.xlarge for GPU for better performance
        endpoint_name=os.environ.get("SAGEMAKER_ENDPOINT_NAME"),
    )


if __name__ == "__main__":
    deploy_endpoint()
