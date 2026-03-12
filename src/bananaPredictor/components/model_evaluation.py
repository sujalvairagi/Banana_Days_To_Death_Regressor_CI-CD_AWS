import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from bananaPredictor import logger
from bananaPredictor.utils.common import save_json
from bananaPredictor.entity.config_entity import EvaluationConfig


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.predictions = None
        self.actuals = None
        self.metrics = {}

    def load_model(self):
        """Load the trained regression model."""
        self.model = tf.keras.models.load_model(
            self.config.trained_model_path,
            compile=False,
        )
        logger.info(f"Loaded trained model from {self.config.trained_model_path}")

    def run_inference(self):
        """Run inference on the test set."""
        img_size = self.config.params_image_size[:2]
        batch_size = self.config.params_batch_size

        test_df = pd.read_csv(Path(self.config.test_data) / "labels.csv")
        test_images_dir = str(Path(self.config.test_data) / "images")

        test_df["days_until_death"] = test_df["days_until_death"].astype(float)

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=test_images_dir,
            x_col="filename",
            y_col="days_until_death",
            target_size=tuple(img_size),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False,
        )

        self.predictions = self.model.predict(test_generator).flatten()
        self.actuals = test_df["days_until_death"].values[: len(self.predictions)]

        logger.info(
            f"Inference complete: {len(self.predictions)} predictions on test set"
        )

    def calculate_metrics(self):
        """Calculate all evaluation metrics."""
        preds = self.predictions
        actuals = self.actuals

        # MAE
        mae = float(np.mean(np.abs(preds - actuals)))

        # RMSE
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        # R² Score
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Within-1-day accuracy
        within_1_day = float(np.mean(np.abs(preds - actuals) <= 1.0))

        # Within-0.5-day accuracy
        within_05_day = float(np.mean(np.abs(preds - actuals) <= 0.5))

        # Direction accuracy (over vs under prediction)
        # True = correct direction or exact
        correct_direction = np.sum(
            ((preds >= actuals) & (actuals >= 0))
            | ((preds <= actuals) & (actuals <= 0))
            | (np.abs(preds - actuals) < 0.5)
        )
        direction_accuracy = float(correct_direction / len(actuals)) if len(actuals) > 0 else 0.0

        # Error by ripeness stage
        error_by_stage = self._calculate_errors_by_stage(preds, actuals)

        self.metrics = {
            "test_mae": round(mae, 4),
            "test_rmse": round(rmse, 4),
            "test_r2": round(r2, 4),
            "within_1_day_accuracy": round(within_1_day, 4),
            "within_0.5_day_accuracy": round(within_05_day, 4),
            "direction_accuracy": round(direction_accuracy, 4),
            "num_test_samples": len(actuals),
            "error_by_stage": error_by_stage,
        }

        logger.info(f"Evaluation Metrics:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Within ±1 day: {within_1_day:.2%}")
        logger.info(f"  Within ±0.5 day: {within_05_day:.2%}")
        logger.info(f"  Direction accuracy: {direction_accuracy:.2%}")

    def _calculate_errors_by_stage(self, preds, actuals):
        """Calculate MAE breakdown by ripeness stage."""
        stages = {
            "overripe_0": (actuals <= 0),
            "spotted_1_3": (actuals > 0) & (actuals <= 3),
            "yellow_4_6": (actuals > 3) & (actuals <= 6),
            "green_7_10": (actuals > 6),
        }

        error_by_stage = {}
        for stage_name, mask in stages.items():
            if np.any(mask):
                stage_mae = float(np.mean(np.abs(preds[mask] - actuals[mask])))
                error_by_stage[stage_name] = {
                    "mae": round(stage_mae, 4),
                    "count": int(np.sum(mask)),
                }

        return error_by_stage

    def log_to_mlflow(self):
        """Log metrics, params, and model to MLflow."""
        import mlflow

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log all params
            all_params = self.config.all_params
            for key, value in all_params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

            # Log metrics
            mlflow.log_metric("test_mae", self.metrics["test_mae"])
            mlflow.log_metric("test_rmse", self.metrics["test_rmse"])
            mlflow.log_metric("test_r2", self.metrics["test_r2"])
            mlflow.log_metric(
                "within_1_day_accuracy", self.metrics["within_1_day_accuracy"]
            )
            mlflow.log_metric(
                "within_0.5_day_accuracy", self.metrics["within_0.5_day_accuracy"]
            )
            mlflow.log_metric(
                "direction_accuracy", self.metrics["direction_accuracy"]
            )

            # Log model
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name="BananaRegressor"
                )
            else:
                mlflow.keras.log_model(self.model, "model")

        logger.info("Metrics and model logged to MLflow")

    def save_metrics(self):
        """Save metrics to JSON files."""
        # Save detailed metrics
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)
        save_json(path=Path(self.config.metrics_path), data=self.metrics)

        # Save scores.json (DVC metric file — flat structure)
        scores = {
            "test_mae": self.metrics["test_mae"],
            "test_rmse": self.metrics["test_rmse"],
            "test_r2": self.metrics["test_r2"],
            "within_1_day_accuracy": self.metrics["within_1_day_accuracy"],
            "within_0.5_day_accuracy": self.metrics["within_0.5_day_accuracy"],
            "direction_accuracy": self.metrics["direction_accuracy"],
        }
        save_json(path=Path(self.config.scores_file), data=scores)
        logger.info(f"Scores saved to {self.config.scores_file}")
