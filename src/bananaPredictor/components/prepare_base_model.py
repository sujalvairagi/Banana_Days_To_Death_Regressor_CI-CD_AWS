import os
import tensorflow as tf
from pathlib import Path
from bananaPredictor import logger
from bananaPredictor.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_efficientnet(self):
        """Load EfficientNet backbone from Keras Applications (ImageNet pretrained)."""
        architecture_map = {
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "EfficientNetB1": tf.keras.applications.EfficientNetB1,
            "EfficientNetB2": tf.keras.applications.EfficientNetB2,
        }

        arch_name = self.config.params_architecture
        if arch_name not in architecture_map:
            raise ValueError(
                f"Unsupported architecture: {arch_name}. "
                f"Choose from: {list(architecture_map.keys())}"
            )

        EfficientNetClass = architecture_map[arch_name]

        self.model = EfficientNetClass(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=tuple(self.config.params_image_size),
        )

        self.model.save(self.config.base_model_path)
        logger.info(
            f"Base {arch_name} model saved to {self.config.base_model_path} | "
            f"Params: {self.model.count_params():,}"
        )

    def _build_regression_head(self, base_output):
        """Build custom regression head on top of backbone features."""
        x = tf.keras.layers.GlobalAveragePooling2D()(base_output)

        head_units = self.config.params_head_units
        dropout_rate = self.config.params_dropout_rate

        # Dense(512) + ReLU + BatchNorm + Dropout
        x = tf.keras.layers.Dense(head_units[0], activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Dense(256) + ReLU + BatchNorm + Dropout
        x = tf.keras.layers.Dense(head_units[1], activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Dense(128) + ReLU
        x = tf.keras.layers.Dense(head_units[2], activation="relu")(x)

        # Dense(1, linear) → Days prediction
        prediction = tf.keras.layers.Dense(1, activation="linear", name="days_prediction")(x)

        return prediction

    def update_base_model(self):
        """Combine backbone + regression head, freeze backbone, save."""
        # Build full model
        base_input = self.model.input
        base_output = self.model.output
        prediction = self._build_regression_head(base_output)

        self.full_model = tf.keras.Model(inputs=base_input, outputs=prediction)

        # Freeze all backbone layers initially
        for layer in self.model.layers:
            layer.trainable = False

        # Compile with initial learning rate
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="huber",
            metrics=["mae"],
        )

        self.full_model.save(self.config.updated_base_model_path)

        logger.info(
            f"Updated model saved to {self.config.updated_base_model_path}"
        )

        # Log architecture summary
        trainable = sum(
            tf.keras.backend.count_params(w) for w in self.full_model.trainable_weights
        )
        non_trainable = sum(
            tf.keras.backend.count_params(w) for w in self.full_model.non_trainable_weights
        )
        logger.info(
            f"Model summary — Trainable: {trainable:,} | "
            f"Non-trainable: {non_trainable:,} | Total: {trainable + non_trainable:,}"
        )

    @staticmethod
    def _freeze_model_layers(model, freeze_till=None):
        """Freeze layers up to freeze_till index. If None, freeze all."""
        if freeze_till is None:
            for layer in model.layers:
                layer.trainable = False
        else:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            for layer in model.layers[-freeze_till:]:
                layer.trainable = True
