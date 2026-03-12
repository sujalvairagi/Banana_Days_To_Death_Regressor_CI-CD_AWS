import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from bananaPredictor import logger


class BananaPredictionPipeline:
    """End-to-end inference: Segmentation → Regression → Annotated output."""

    def __init__(self):
        self.seg_model = None
        self.reg_model = None
        self._load_models()

    def _load_models(self):
        """Load both segmentation and regression models."""
        from ultralytics import YOLO

        # Load segmentation model (YOLOv8 weights downloaded from Roboflow)
        seg_model_path = Path("model/segmentation_model/weights/best.pt")
        if not seg_model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at: {seg_model_path}\n"
                "Download best.pt from Roboflow and place it at: "
                "model/segmentation_model/weights/best.pt"
            )
        self.seg_model = YOLO(str(seg_model_path))
        logger.info(f"Segmentation model loaded from {seg_model_path}")

        # Load regression model
        reg_model_path = Path("model/regression_model.h5")
        if not reg_model_path.exists():
            # Fallback: try artifacts path
            reg_model_path = Path("artifacts/training/regression_model.h5")
        if not reg_model_path.exists():
            raise FileNotFoundError(
                "Regression model not found. Run the training pipeline first (python main.py)"
            )
        self.reg_model = tf.keras.models.load_model(
            str(reg_model_path), compile=False
        )
        logger.info(f"Regression model loaded from {reg_model_path}")

    def predict(self, image_path):
        """
        Full prediction pipeline.

        Args:
            image_path (str): Path to input image with banana(s)

        Returns:
            dict with total_bananas, predictions list, summary, and annotated_image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Step 1: Run segmentation
        seg_results = self.seg_model.predict(source=image_path, conf=0.5, verbose=False)

        predictions = []
        banana_id = 0

        for result in seg_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                banana_id += 1
                x1, y1, x2, y2 = map(int, box)

                # Step 2: Crop individual banana
                crop = self._crop_banana(image, x1, y1, x2, y2)

                # Step 3: Predict days-to-death for this banana
                days_left = self._predict_days(crop)

                # Step 4: Categorize
                category = self._categorize(days_left)

                predictions.append({
                    "banana_id": banana_id,
                    "bbox": [x1, y1, x2, y2],
                    "days_left": round(float(days_left), 2),
                    "segmentation_confidence": round(float(conf), 4),
                    "category": category,
                })

        # Step 5: Generate summary
        summary = self._calculate_summary(predictions)

        # Step 6: Create annotated image
        annotated_image = self._annotate_image(image, predictions)

        return {
            "total_bananas": len(predictions),
            "predictions": predictions,
            "summary": summary,
            "annotated_image": annotated_image,
        }

    def _crop_banana(self, image, x1, y1, x2, y2):
        """Crop a banana region from the image with padding."""
        h, w = image.shape[:2]

        # Add small padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]
        return crop

    def _predict_days(self, crop):
        """Predict days-to-death for a single banana crop."""
        # Preprocess: resize to model input size
        img = cv2.resize(crop, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = self.reg_model.predict(img, verbose=0)
        days = float(prediction[0][0])

        # Clamp to reasonable range
        days = max(0.0, days)

        return days

    def _categorize(self, days_left):
        """Categorize banana ripeness based on predicted days."""
        if days_left <= 0:
            return "overripe"
        elif days_left <= 3:
            return "spotted"
        elif days_left <= 6:
            return "yellow"
        else:
            return "green"

    def _calculate_summary(self, predictions):
        """Calculate aggregate statistics."""
        if not predictions:
            return {
                "avg_days": 0,
                "fresh_count": 0,
                "ripe_count": 0,
                "overripe_count": 0,
            }

        days_list = [p["days_left"] for p in predictions]
        categories = [p["category"] for p in predictions]

        return {
            "avg_days": round(float(np.mean(days_list)), 2),
            "min_days": round(float(np.min(days_list)), 2),
            "max_days": round(float(np.max(days_list)), 2),
            "fresh_count": categories.count("green"),
            "ripe_count": categories.count("yellow") + categories.count("spotted"),
            "overripe_count": categories.count("overripe"),
        }

    def _annotate_image(self, image, predictions):
        """Draw bounding boxes and labels on the image."""
        annotated = image.copy()

        color_map = {
            "green": (0, 200, 0),
            "yellow": (0, 255, 255),
            "spotted": (0, 165, 255),
            "overripe": (0, 0, 255),
        }

        for pred in predictions:
            x1, y1, x2, y2 = pred["bbox"]
            color = color_map.get(pred["category"], (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"#{pred['banana_id']}: {pred['days_left']:.1f}d ({pred['category']})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Background for text
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return annotated
