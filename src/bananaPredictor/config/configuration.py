from bananaPredictor.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from bananaPredictor.utils.common import read_yaml, create_directories
from bananaPredictor.entity.config_entity import (
    DataIngestionConfig,
    SegDataPreparationConfig,
    SegDataSplitterConfig,
    SegModelTrainingConfig,
    SegmentationValidationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            dataset_dir=Path(config.dataset_dir),
            train_dir=Path(config.train_dir),
            val_dir=Path(config.val_dir),
            test_dir=Path(config.test_dir),
            split_ratio=dict(config.split_ratio),
        )

        return data_ingestion_config

    def get_seg_data_preparation_config(self) -> SegDataPreparationConfig:
        config = self.config.seg_data_preparation

        create_directories([config.root_dir, config.raw_data_dir])

        seg_data_config = SegDataPreparationConfig(
            root_dir=Path(config.root_dir),
            raw_data_dir=Path(config.raw_data_dir),
            coco_annotation_file=Path(config.coco_annotation_file),
            yolo_output_dir=Path(config.yolo_output_dir),
            yolo_images_dir=Path(config.yolo_images_dir),
            yolo_labels_dir=Path(config.yolo_labels_dir),
        )

        return seg_data_config

    def get_seg_data_splitter_config(self) -> SegDataSplitterConfig:
        config = self.config.seg_data_splitter

        create_directories([config.root_dir])

        seg_split_config = SegDataSplitterConfig(
            root_dir=Path(config.root_dir),
            source_images_dir=Path(config.source_images_dir),
            source_labels_dir=Path(config.source_labels_dir),
            train_dir=Path(config.train_dir),
            val_dir=Path(config.val_dir),
            test_dir=Path(config.test_dir),
            dataset_yaml_path=Path(config.dataset_yaml_path),
            split_ratio=dict(config.split_ratio),
        )

        return seg_split_config

    def get_seg_model_training_config(self) -> SegModelTrainingConfig:
        config = self.config.seg_model_training

        create_directories([config.root_dir, config.trained_weights_dir])

        seg_train_config = SegModelTrainingConfig(
            root_dir=Path(config.root_dir),
            base_model=config.base_model,
            dataset_yaml_path=Path(config.dataset_yaml_path),
            trained_weights_dir=Path(config.trained_weights_dir),
            best_weights_path=Path(config.best_weights_path),
            export_weights_path=Path(config.export_weights_path),
            training_results_dir=Path(config.training_results_dir),
            params_epochs=self.params.SEG_EPOCHS,
            params_image_size=self.params.SEG_IMAGE_SIZE,
            params_batch_size=self.params.SEG_BATCH_SIZE,
            params_lr=self.params.SEG_LR,
            params_patience=self.params.SEG_PATIENCE,
        )

        return seg_train_config

    def get_segmentation_validation_config(self) -> SegmentationValidationConfig:
        config = self.config.segmentation_validation

        create_directories([config.root_dir])

        seg_val_config = SegmentationValidationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            confidence_threshold=float(config.confidence_threshold),
            validation_data=Path(config.validation_data),
            report_path=Path(config.report_path),
        )

        return seg_val_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_weights=self.params.WEIGHTS,
            params_include_top=self.params.INCLUDE_TOP,
            params_architecture=self.params.ARCHITECTURE,
            params_head_units=self.params.HEAD_UNITS,
            params_dropout_rate=self.params.DROPOUT_RATE,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        prepare_base_model = self.config.prepare_base_model

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(config.training_data),
            validation_data=Path(config.validation_data),
            history_path=Path(config.history_path),
            mlflow_uri=config.mlflow_uri,
            params_epochs_phase_1=self.params.PHASE_1_EPOCHS,
            params_epochs_phase_2=self.params.PHASE_2_EPOCHS,
            params_epochs_phase_3=self.params.PHASE_3_EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE,
            params_lr_phase_1=self.params.PHASE_1_LR,
            params_lr_phase_2=self.params.PHASE_2_LR,
            params_lr_phase_3=self.params.PHASE_3_LR,
            params_unfreeze_layers=self.params.PHASE_2_UNFREEZE_LAYERS,
            params_augmentation=self.params.AUGMENTATION,
            params_early_stopping_patience=self.params.PHASE_3_EARLY_STOPPING_PATIENCE,
            params_huber_delta=self.params.HUBER_DELTA,
            params_ordinal_weight=self.params.ORDINAL_WEIGHT,
            all_params=dict(self.params),
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation

        create_directories([config.root_dir])

        eval_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data=Path(config.test_data),
            trained_model_path=Path(config.trained_model_path),
            metrics_path=Path(config.metrics_path),
            scores_file=Path(config.scores_file),
            mlflow_uri=config.mlflow_uri,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            all_params=dict(self.params),
        )

        return eval_config
