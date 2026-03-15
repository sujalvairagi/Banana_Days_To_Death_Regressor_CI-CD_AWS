import os
import json
import shutil
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.entity.config_entity import SegDataPreparationConfig


class SegDataPreparation:
    """Convert COCO-format segmentation dataset (from Roboflow) to YOLO format."""

    def __init__(self, config: SegDataPreparationConfig):
        self.config = config
        self.coco_data = None
        self.category_map = {}

    def validate_coco_dataset(self):
        """Validate that the raw COCO dataset exists with required files."""
        raw_dir = Path(self.config.raw_data_dir)
        annotation_file = Path(self.config.coco_annotation_file)

        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw segmentation data directory not found at: {raw_dir}\n"
                "Please download your COCO segmentation dataset from Roboflow:\n"
                "  1. Go to your Roboflow project → Dataset → Download\n"
                "  2. Select 'COCO Segmentation' format\n"
                "  3. Unzip and place contents in: artifacts/seg_data/raw/\n"
                "     Expected structure:\n"
                "       artifacts/seg_data/raw/\n"
                "       ├── _annotations.coco.json\n"
                "       └── *.jpg / *.png (image files)"
            )

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found at: {annotation_file}\n"
                "Expected '_annotations.coco.json' in artifacts/seg_data/raw/"
            )

        # Load and validate COCO JSON
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        required_keys = ["images", "annotations", "categories"]
        for key in required_keys:
            if key not in self.coco_data:
                raise ValueError(
                    f"Invalid COCO annotation file: missing '{key}' key"
                )

        num_images = len(self.coco_data["images"])
        num_annotations = len(self.coco_data["annotations"])
        num_categories = len(self.coco_data["categories"])

        logger.info(
            f"COCO dataset validated: {num_images} images, "
            f"{num_annotations} annotations, {num_categories} categories"
        )

        # Build category ID to index mapping (YOLO uses 0-indexed class IDs)
        self.category_map = {}
        for idx, cat in enumerate(self.coco_data["categories"]):
            self.category_map[cat["id"]] = idx
            logger.info(f"  Category: {cat['name']} (COCO id={cat['id']} → YOLO class={idx})")

    def convert_coco_to_yolo(self):
        """Convert COCO annotations to YOLO segmentation format."""
        if self.coco_data is None:
            raise RuntimeError("Call validate_coco_dataset() before conversion")

        yolo_images_dir = Path(self.config.yolo_images_dir)
        yolo_labels_dir = Path(self.config.yolo_labels_dir)
        raw_dir = Path(self.config.raw_data_dir)

        os.makedirs(yolo_images_dir, exist_ok=True)
        os.makedirs(yolo_labels_dir, exist_ok=True)

        # Build image ID → image info lookup
        image_lookup = {}
        for img_info in self.coco_data["images"]:
            image_lookup[img_info["id"]] = img_info

        # Group annotations by image ID
        annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        converted_count = 0
        skipped_count = 0

        for img_id, img_info in image_lookup.items():
            filename = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # Copy image to YOLO images directory
            src_img = raw_dir / filename
            if not src_img.exists():
                logger.warning(f"Image not found: {src_img}, skipping")
                skipped_count += 1
                continue

            dst_img = yolo_images_dir / filename
            shutil.copy2(src_img, dst_img)

            # Generate YOLO label file
            label_filename = Path(filename).stem + ".txt"
            label_path = yolo_labels_dir / label_filename

            annotations = annotations_by_image.get(img_id, [])

            with open(label_path, "w") as label_file:
                for ann in annotations:
                    # Get YOLO class index
                    class_idx = self.category_map.get(ann["category_id"])
                    if class_idx is None:
                        continue

                    # Convert segmentation polygons to normalized YOLO format
                    if "segmentation" in ann and ann["segmentation"]:
                        for seg_polygon in ann["segmentation"]:
                            if isinstance(seg_polygon, list) and len(seg_polygon) >= 6:
                                # Normalize polygon coordinates
                                normalized_points = []
                                for i in range(0, len(seg_polygon), 2):
                                    x = seg_polygon[i] / img_width
                                    y = seg_polygon[i + 1] / img_height
                                    # Clamp to [0, 1]
                                    x = max(0.0, min(1.0, x))
                                    y = max(0.0, min(1.0, y))
                                    normalized_points.extend([x, y])

                                # YOLO seg format: class_id x1 y1 x2 y2 ... xn yn
                                points_str = " ".join(
                                    f"{p:.6f}" for p in normalized_points
                                )
                                label_file.write(f"{class_idx} {points_str}\n")

            converted_count += 1

        logger.info(
            f"COCO → YOLO conversion complete: {converted_count} images converted, "
            f"{skipped_count} skipped"
        )

        # Save class names for later use in data.yaml
        class_names = [cat["name"] for cat in self.coco_data["categories"]]
        class_names_path = Path(self.config.yolo_output_dir) / "class_names.json"
        with open(class_names_path, "w") as f:
            json.dump(class_names, f, indent=2)

        logger.info(f"Class names saved to {class_names_path}")
