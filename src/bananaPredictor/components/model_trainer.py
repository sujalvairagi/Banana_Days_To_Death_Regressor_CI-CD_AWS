import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.entity.config_entity import TrainingConfig


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.history = {"phase_1": None, "phase_2": None, "phase_3": None}

    def load_base_model(self):
        """Load the updated base model from Stage 3."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False,
        )
        logger.info(f"Loaded base model from {self.config.updated_base_model_path}")

    def _combined_loss(self, y_true, y_pred):
        """Custom loss: Huber + Ordinal penalty.

        Huber loss is robust to labeling noise.
        Ordinal penalty penalizes wrong-direction errors more heavily.
        """
        huber_delta = self.config.params_huber_delta
        ordinal_weight = self.config.params_ordinal_weight

        # Huber loss
        huber = tf.keras.losses.Huber(delta=huber_delta)(y_true, y_pred)

        # Ordinal penalty: penalize wrong direction more
        direction_error = tf.abs(
            tf.sign(y_pred - y_true) * tf.square(y_pred - y_true)
        )
        ordinal = tf.reduce_mean(direction_error)

        return (1.0 - ordinal_weight) * huber + ordinal_weight * ordinal

    def prepare_data_generators(self):
        """Create training and validation data generators."""
        img_size = self.config.params_image_size[:2]  # [224, 224]
        batch_size = self.config.params_batch_size

        # Training augmentation
        if self.config.params_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=15,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                zoom_range=0.1,
                fill_mode="nearest",
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
            )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
        )

        # Load labels
        train_df = pd.read_csv(Path(self.config.training_data) / "labels.csv")
        val_df = pd.read_csv(Path(self.config.validation_data) / "labels.csv")

        # Ensure 'days_until_death' column is string for flow_from_dataframe
        train_df["days_until_death"] = train_df["days_until_death"].astype(float)
        val_df["days_until_death"] = val_df["days_until_death"].astype(float)

        train_images_dir = str(Path(self.config.training_data) / "images")
        val_images_dir = str(Path(self.config.validation_data) / "images")

        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=train_images_dir,
            x_col="filename",
            y_col="days_until_death",
            target_size=tuple(img_size),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=True,
        )

        self.val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=val_images_dir,
            x_col="filename",
            y_col="days_until_death",
            target_size=tuple(img_size),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False,
        )

        logger.info(
            f"Data generators ready — Train: {len(train_df)} samples, "
            f"Val: {len(val_df)} samples"
        )

    def _get_callbacks(self, phase_name):
        """Get callbacks for a training phase."""
        callbacks = []

        # Model checkpoint — save best model
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

        # Early stopping (only for phase 3)
        if phase_name == "phase_3":
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.params_early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        # Reduce LR on Plateau
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            )
        )

        # TensorBoard
        log_dir = os.path.join(self.config.root_dir, "tensorboard_logs", phase_name)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        )

        return callbacks

    def train(self):
        """Execute 3-phase training schedule."""
        # ===== PHASE 1: Freeze backbone, train head only =====
        logger.info("=" * 60)
        logger.info("PHASE 1: Training regression head only (backbone frozen)")
        logger.info("=" * 60)

        # Freeze all backbone layers (EfficientNet layers)
        for layer in self.model.layers:
            if "efficientnet" in layer.name.lower() or not isinstance(
                layer, (tf.keras.layers.Dense, tf.keras.layers.BatchNormalization,
                        tf.keras.layers.Dropout, tf.keras.layers.GlobalAveragePooling2D)
            ):
                layer.trainable = False

        # Ensure head layers are trainable
        head_layer_types = (
            tf.keras.layers.Dense,
            tf.keras.layers.BatchNormalization,
            tf.keras.layers.Dropout,
            tf.keras.layers.GlobalAveragePooling2D,
        )
        for layer in self.model.layers:
            if isinstance(layer, head_layer_types):
                layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_lr_phase_1),
            loss=self._combined_loss,
            metrics=["mae"],
        )

        self._log_trainable_summary("Phase 1")

        self.history["phase_1"] = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs_phase_1,
            validation_data=self.val_generator,
            callbacks=self._get_callbacks("phase_1"),
        )

        # ===== PHASE 2: Partial unfreeze =====
        logger.info("=" * 60)
        logger.info(
            f"PHASE 2: Unfreezing last {self.config.params_unfreeze_layers} layers"
        )
        logger.info("=" * 60)

        # Unfreeze last N layers of the backbone
        total_layers = len(self.model.layers)
        for layer in self.model.layers[-(self.config.params_unfreeze_layers):]:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_lr_phase_2),
            loss=self._combined_loss,
            metrics=["mae"],
        )

        self._log_trainable_summary("Phase 2")

        self.history["phase_2"] = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs_phase_2,
            validation_data=self.val_generator,
            callbacks=self._get_callbacks("phase_2"),
        )

        # ===== PHASE 3: Full fine-tune =====
        logger.info("=" * 60)
        logger.info("PHASE 3: Full fine-tuning (all layers trainable)")
        logger.info("=" * 60)

        for layer in self.model.layers:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_lr_phase_3),
            loss=self._combined_loss,
            metrics=["mae"],
        )

        self._log_trainable_summary("Phase 3")

        self.history["phase_3"] = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs_phase_3,
            validation_data=self.val_generator,
            callbacks=self._get_callbacks("phase_3"),
        )

        logger.info("All 3 training phases completed.")

    def _log_trainable_summary(self, phase_name):
        """Log count of trainable vs non-trainable params."""
        trainable = sum(
            tf.keras.backend.count_params(w) for w in self.model.trainable_weights
        )
        non_trainable = sum(
            tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights
        )
        logger.info(
            f"{phase_name} — Trainable: {trainable:,} | "
            f"Non-trainable: {non_trainable:,}"
        )

    def save_model(self):
        """Save the final trained model and training history."""
        # Save model
        self.model.save(self.config.trained_model_path)
        logger.info(f"Trained model saved to {self.config.trained_model_path}")

        # Also copy to model/ for deployment
        deploy_path = Path("model") / "regression_model.h5"
        os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
        self.model.save(str(deploy_path))
        logger.info(f"Deployment model copied to {deploy_path}")

        # Save training history
        history_data = {}
        for phase_name, hist in self.history.items():
            if hist is not None:
                history_data[phase_name] = {
                    key: [float(v) for v in values]
                    for key, values in hist.history.items()
                }

        with open(self.config.history_path, "w") as f:
            json.dump(history_data, f, indent=4)

        logger.info(f"Training history saved to {self.config.history_path}")
