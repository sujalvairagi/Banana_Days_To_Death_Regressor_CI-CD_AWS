from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    split_ratio: dict


@dataclass(frozen=True)
class SegDataPreparationConfig:
    root_dir: Path
    raw_data_dir: Path
    coco_annotation_file: Path
    yolo_output_dir: Path
    yolo_images_dir: Path
    yolo_labels_dir: Path


@dataclass(frozen=True)
class SegDataSplitterConfig:
    root_dir: Path
    source_images_dir: Path
    source_labels_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    dataset_yaml_path: Path
    split_ratio: dict


@dataclass(frozen=True)
class SegModelTrainingConfig:
    root_dir: Path
    base_model: str
    dataset_yaml_path: Path
    trained_weights_dir: Path
    best_weights_path: Path
    export_weights_path: Path
    training_results_dir: Path
    params_epochs: int
    params_image_size: int
    params_batch_size: int
    params_lr: float
    params_patience: int


@dataclass(frozen=True)
class SegmentationValidationConfig:
    root_dir: Path
    model_path: Path
    confidence_threshold: float
    validation_data: Path
    report_path: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_weights: str
    params_include_top: bool
    params_architecture: str
    params_head_units: list
    params_dropout_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    history_path: Path
    mlflow_uri: str
    params_epochs_phase_1: int
    params_epochs_phase_2: int
    params_epochs_phase_3: int
    params_batch_size: int
    params_image_size: list
    params_lr_phase_1: float
    params_lr_phase_2: float
    params_lr_phase_3: float
    params_unfreeze_layers: int
    params_augmentation: bool
    params_early_stopping_patience: int
    params_huber_delta: float
    params_ordinal_weight: float
    all_params: dict


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    test_data: Path
    trained_model_path: Path
    metrics_path: Path
    scores_file: Path
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    all_params: dict
