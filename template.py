import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "bananaPredictor"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/segmentation_validator.py",
    f"src/{project_name}/components/prepare_base_model.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_segmentation_validation.py",
    f"src/{project_name}/pipeline/stage_03_prepare_base_model.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/prediction.py",
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    "research/01_data_ingestion.ipynb",
    "research/02_segmentation_testing.ipynb",
    "research/03_prepare_base_model.ipynb",
    "research/04_model_training.ipynb",
    "research/05_evaluation.ipynb",
    "model/segmentation_model/weights/.gitkeep",
    "artifacts/.gitkeep",
    "logs/.gitkeep",
    "mlruns/.gitkeep",
    "templates/index.html",
    "main.py",
    "app.py",
    "scores.json",
    "setup.py",
    "requirements.txt",
    "Dockerfile",
    ".github/workflows/main.yaml",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
