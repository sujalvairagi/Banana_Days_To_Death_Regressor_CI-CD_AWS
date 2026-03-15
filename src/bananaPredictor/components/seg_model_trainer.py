import os
import shutil
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.entity.config_entity import SegModelTrainingConfig


class SegModelTrainer:
    """Train YOLOv8 segmentation model on the prepared dataset."""

    def __init__(self, config: SegModelTrainingConfig):
        self.config = config
        self.model = None
        self.results = None

    def train(self):
        """Train YOLOv8-seg model using ultralytics API."""
        from ultralytics import YOLO

        dataset_yaml = Path(self.config.dataset_yaml_path)
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found at: {dataset_yaml}\n"
                "Run the segmentation data splitter stage first."
            )

        logger.info(f"Loading base model: {self.config.base_model}")
        self.model = YOLO(self.config.base_model)

        logger.info("Starting YOLOv8 segmentation training...")
        logger.info(f"  Dataset: {dataset_yaml}")
        logger.info(f"  Epochs: {self.config.params_epochs}")
        logger.info(f"  Image size: {self.config.params_image_size}")
        logger.info(f"  Batch size: {self.config.params_batch_size}")
        logger.info(f"  Learning rate: {self.config.params_lr}")
        logger.info(f"  Patience: {self.config.params_patience}")

        results_dir = Path(self.config.training_results_dir)

        self.results = self.model.train(
            data=str(dataset_yaml),
            epochs=self.config.params_epochs,
            imgsz=self.config.params_image_size,
            batch=self.config.params_batch_size,
            lr0=self.config.params_lr,
            patience=self.config.params_patience,
            project=str(results_dir.parent),
            name=results_dir.name,
            exist_ok=True,
            verbose=True,
            save=True,
            plots=True,
        )

        logger.info("YOLOv8 segmentation training completed.")

    def export_best_weights(self):
        """Copy best weights to the model deployment directory."""
        # YOLOv8 saves best weights in the training results directory
        results_dir = Path(self.config.training_results_dir)
        yolo_best_weights = results_dir / "weights" / "best.pt"

        if not yolo_best_weights.exists():
            # Fallback: check if weights are in the results dir directly
            yolo_best_weights = results_dir / "best.pt"

        if not yolo_best_weights.exists():
            raise FileNotFoundError(
                f"Trained best weights not found at expected locations:\n"
                f"  - {results_dir / 'weights' / 'best.pt'}\n"
                f"  - {results_dir / 'best.pt'}\n"
                "Training may have failed."
            )

        # Copy to the configured weights path in artifacts
        best_weights_dest = Path(self.config.best_weights_path)
        os.makedirs(best_weights_dest.parent, exist_ok=True)
        shutil.copy2(yolo_best_weights, best_weights_dest)
        logger.info(f"Best weights saved to: {best_weights_dest}")

        # Copy to the model deployment directory
        export_dest = Path(self.config.export_weights_path)
        os.makedirs(export_dest.parent, exist_ok=True)
        shutil.copy2(yolo_best_weights, export_dest)
        logger.info(f"Best weights exported to: {export_dest}")

        # Also copy last.pt if it exists
        yolo_last_weights = results_dir / "weights" / "last.pt"
        if yolo_last_weights.exists():
            last_dest = Path(self.config.trained_weights_dir) / "last.pt"
            shutil.copy2(yolo_last_weights, last_dest)
            logger.info(f"Last weights saved to: {last_dest}")
























































