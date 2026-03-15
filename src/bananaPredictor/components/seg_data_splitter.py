import os
import json
import shutil
import yaml
import numpy as np
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.entity.config_entity import SegDataSplitterConfig


class SegDataSplitter:
    """Split YOLO-format segmentation dataset into train/val/test and generate data.yaml."""

    def __init__(self, config: SegDataSplitterConfig):
        self.config = config
        self.class_names = []

    def split_dataset(self):
        """Split images and labels into train/val/test directories."""
        images_dir = Path(self.config.source_images_dir)
        labels_dir = Path(self.config.source_labels_dir)

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Source images directory not found at: {images_dir}\n"
                "Run the segmentation data preparation stage first."
            )
        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Source labels directory not found at: {labels_dir}\n"
                "Run the segmentation data preparation stage first."
            )

        # Collect all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = sorted([
            f.name for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            raise ValueError(f"No images found in {images_dir}")

        logger.info(f"Found {len(image_files)} images for splitting")

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        rng.shuffle(image_files)

        # Calculate split indices
        n = len(image_files)
        train_ratio = self.config.split_ratio["train"]
        val_ratio = self.config.split_ratio["val"]

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": (self.config.train_dir, image_files[:train_end]),
            "val": (self.config.val_dir, image_files[train_end:val_end]),
            "test": (self.config.test_dir, image_files[val_end:]),
        }

        for split_name, (split_dir, split_files) in splits.items():
            split_dir = Path(split_dir)
            split_images_dir = split_dir / "images"
            split_labels_dir = split_dir / "labels"
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)

            for img_filename in split_files:
                # Copy image
                src_img = images_dir / img_filename
                dst_img = split_images_dir / img_filename
                shutil.copy2(src_img, dst_img)

                # Copy corresponding label file
                label_filename = Path(img_filename).stem + ".txt"
                src_label = labels_dir / label_filename
                dst_label = split_labels_dir / label_filename
                if src_label.exists():
                    shutil.copy2(src_label, dst_label)
                else:
                    # Create empty label file for images with no annotations
                    dst_label.touch()

            logger.info(
                f"Split '{split_name}': {len(split_files)} images → {split_dir}"
            )

        logger.info("Dataset splitting completed successfully.")

    def create_dataset_yaml(self):
        """Generate data.yaml required by YOLOv8 for training."""
        # Load class names from the preparation stage output
        class_names_path = Path(self.config.source_images_dir).parent / "class_names.json"
        if class_names_path.exists():
            with open(class_names_path, "r") as f:
                self.class_names = json.load(f)
        else:
            logger.warning(
                f"class_names.json not found at {class_names_path}. "
                "Using default class name 'banana'."
            )
            self.class_names = ["banana"]

        # Build data.yaml with absolute paths for YOLOv8
        train_path = str(Path(self.config.train_dir).resolve())
        val_path = str(Path(self.config.val_dir).resolve())
        test_path = str(Path(self.config.test_dir).resolve())

        data_yaml = {
            "path": str(Path(self.config.root_dir).resolve()),
            "train": os.path.join(train_path, "images"),
            "val": os.path.join(val_path, "images"),
            "test": os.path.join(test_path, "images"),
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        yaml_path = Path(self.config.dataset_yaml_path)
        os.makedirs(yaml_path.parent, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Dataset YAML saved to {yaml_path}")
        logger.info(f"  Classes ({len(self.class_names)}): {self.class_names}")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Val: {val_path}")
        logger.info(f"  Test: {test_path}")
