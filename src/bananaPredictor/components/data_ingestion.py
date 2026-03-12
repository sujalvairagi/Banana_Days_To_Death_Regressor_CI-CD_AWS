import os
import zipfile
import shutil
import gdown
import pandas as pd
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.utils.common import get_size
from bananaPredictor.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        """Download dataset zip from Google Drive."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading dataset from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, str(zip_download_dir))

            logger.info(f"Downloaded dataset to {zip_download_dir}")

        except Exception as e:
            raise e

    def extract_zip(self):
        """Extract the downloaded zip file."""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Extracted zip file to {unzip_path}")

    def split_dataset(self):
        """Split dataset into train/val/test preserving banana_id groups."""
        dataset_dir = self.config.dataset_dir
        labels_path = os.path.join(dataset_dir, "labels.csv")

        if not os.path.exists(labels_path):
            logger.warning(
                f"labels.csv not found at {labels_path}. "
                "Skipping split — data may already be split or uses a different format."
            )
            return

        logger.info("Starting dataset split...")
        df = pd.read_csv(labels_path)

        # Split by banana_id to keep all images of one banana in the same set
        if "banana_id" in df.columns:
            unique_ids = df["banana_id"].unique()
        else:
            # Fallback: treat each image as independent
            unique_ids = df["filename"].unique()

        # Shuffle with fixed seed for reproducibility
        import numpy as np
        rng = np.random.RandomState(42)
        rng.shuffle(unique_ids)

        n = len(unique_ids)
        train_ratio = self.config.split_ratio["train"]
        val_ratio = self.config.split_ratio["val"]

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_ids = set(unique_ids[:train_end])
        val_ids = set(unique_ids[train_end:val_end])
        test_ids = set(unique_ids[val_end:])

        id_col = "banana_id" if "banana_id" in df.columns else "filename"

        splits = {
            "train": (self.config.train_dir, train_ids),
            "val": (self.config.val_dir, val_ids),
            "test": (self.config.test_dir, test_ids),
        }

        images_dir = os.path.join(dataset_dir, "images")

        for split_name, (split_dir, split_ids) in splits.items():
            split_dir = Path(split_dir)
            images_split_dir = split_dir / "images"
            os.makedirs(images_split_dir, exist_ok=True)

            split_df = df[df[id_col].isin(split_ids)]
            split_df.to_csv(split_dir / "labels.csv", index=False)

            # Copy images to split directory
            for filename in split_df["filename"].values:
                src = os.path.join(images_dir, filename)
                dst = images_split_dir / filename
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            logger.info(
                f"Split '{split_name}': {len(split_df)} samples, "
                f"{len(split_ids)} unique IDs"
            )

        logger.info("Dataset split completed successfully.")
