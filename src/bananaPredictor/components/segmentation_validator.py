import os
import time
import numpy as np
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.utils.common import save_json
from bananaPredictor.entity.config_entity import SegmentationValidationConfig


class SegmentationValidator:
    def __init__(self, config: SegmentationValidationConfig):
        self.config = config
        self.model = None
        self.results = {}

    def load_model(self):
        """Load YOLOv8 segmentation model from local weights (downloaded from Roboflow)."""
        from ultralytics import YOLO

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at: {model_path}\n"
                "Please download your trained YOLOv8 model from Roboflow:\n"
                "  1. Go to your Roboflow project → Trained Model\n"
                "  2. Click 'Download' → select 'YOLOv8' format\n"
                "  3. Place the 'best.pt' file at: model/segmentation_model/weights/best.pt"
            )

        self.model = YOLO(str(model_path))
        logger.info(f"Loaded YOLOv8 segmentation model from {model_path}")

    def run_validation(self):
        """Run segmentation model on validation images and compute metrics."""
        val_data_path = Path(self.config.validation_data)
        images_dir = val_data_path / "images"

        if not images_dir.exists():
            logger.warning(f"Validation images directory not found at {images_dir}")
            images_dir = val_data_path  # fallback: treat val_data as images dir

        # Collect all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning("No images found in validation directory")
            self.results = {"error": "No validation images found"}
            return

        logger.info(f"Running validation on {len(image_files)} images...")

        total_detections = 0
        total_confidence = 0.0
        inference_times = []

        for img_path in image_files:
            start_time = time.time()
            results = self.model.predict(
                source=str(img_path),
                conf=self.config.confidence_threshold,
                verbose=False,
            )
            elapsed = (time.time() - start_time) * 1000  # ms

            inference_times.append(elapsed)

            for result in results:
                if result.boxes is not None:
                    num_boxes = len(result.boxes)
                    total_detections += num_boxes
                    if num_boxes > 0:
                        total_confidence += float(result.boxes.conf.sum())

        avg_confidence = (
            total_confidence / total_detections if total_detections > 0 else 0.0
        )
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0

        self.results = {
            "total_images": len(image_files),
            "total_detections": total_detections,
            "avg_detections_per_image": round(
                total_detections / len(image_files), 2
            ),
            "avg_confidence": round(avg_confidence, 4),
            "avg_inference_time_ms": round(avg_inference_time, 2),
            "confidence_threshold": self.config.confidence_threshold,
            "model_path": str(self.config.model_path),
        }

        logger.info(
            f"Validation complete: {total_detections} detections across "
            f"{len(image_files)} images, avg confidence: {avg_confidence:.4f}"
        )

        # Run built-in YOLOv8 val if a YAML dataset config exists
        try:
            val_results = self.model.val(verbose=False)
            if val_results is not None:
                # Extract official metrics if available
                metrics = val_results.results_dict if hasattr(val_results, 'results_dict') else {}
                if metrics:
                    self.results.update({
                        "mAP_50": round(metrics.get("metrics/mAP50(B)", 0.0), 4),
                        "mAP_50_95": round(metrics.get("metrics/mAP50-95(B)", 0.0), 4),
                        "precision": round(metrics.get("metrics/precision(B)", 0.0), 4),
                        "recall": round(metrics.get("metrics/recall(B)", 0.0), 4),
                    })
                    # Calculate F1 from precision and recall
                    p = self.results.get("precision", 0)
                    r = self.results.get("recall", 0)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    self.results["f1_score"] = round(f1, 4)
        except Exception as e:
            logger.info(
                f"Built-in YOLOv8 val() not available (no dataset YAML): {e}. "
                "Using prediction-based metrics only."
            )

    def save_metrics(self):
        """Save validation metrics to JSON report."""
        os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
        save_json(path=Path(self.config.report_path), data=self.results)
        logger.info(f"Segmentation validation report saved to {self.config.report_path}")
