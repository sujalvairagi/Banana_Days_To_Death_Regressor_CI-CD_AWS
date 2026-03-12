# BANANA DAYS-TO-DEATH PREDICTION — PROJECT BLUEPRINT

> **Project Goal**: Predict how many days until each banana in an image becomes inedible  
> **Approach**: Instance Segmentation → Regression (per-banana predictions)  
> **Architecture**: Modular, Reproducible, DVC-enabled, GPU-based Training Pipeline

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Directory Tree](#2-directory-tree)
3. [Architecture Flow](#3-architecture-flow)
4. [Pipeline Stages Breakdown](#4-pipeline-stages-breakdown)
5. [Configuration Files](#5-configuration-files)
6. [Entity Definitions](#6-entity-definitions)
7. [Components Deep Dive](#7-components-deep-dive)
8. [DVC Pipeline](#8-dvc-pipeline)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Deployment Strategy](#10-deployment-strategy)
11. [Innovation Points](#11-innovation-points)

---

## 1. PROJECT OVERVIEW

### Problem Statement
When a vendor/buyer photographs banana bunches at a market, the system should:
1. Detect each individual banana in the image
2. Predict days until inedible for each banana
3. Display bounding boxes with predictions

### Technical Approach

```
┌────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE                        │
│                                                             │
│  Input Image (Multiple Bananas)                            │
│         ↓                                                   │
│  Stage 1: Instance Segmentation (YOLOv8 - Roboflow)        │
│         ↓                                                   │
│  Cropped Individual Bananas (N crops)                      │
│         ↓                                                   │
│  Stage 2: Days-to-Death Regression (EfficientNet)          │
│         ↓                                                   │
│  Per-Banana Predictions: [3.2 days, 2.1 days, 5.4 days...] │
│         ↓                                                   │
│  Visualization: Bounding boxes + labels                     │
└────────────────────────────────────────────────────────────┘
```

### Key Features
- **Modular**: Each stage is independent, replaceable
- **Reproducible**: DVC tracks data, models, metrics
- **Parameter-driven**: All hyperparams in `params.yaml`
- **Production-ready**: Flask API, Docker deployment
- **Experiment tracking**: MLflow integration

---

## 2. DIRECTORY TREE

```
banana-days-predictor/
│
├── .github/
│   └── workflows/
│       └── main.yaml                      # CI/CD: GitHub Actions → AWS/Azure
│
├── config/
│   └── config.yaml                        # Paths, URLs, directories
│
├── src/
│   └── bananaPredictor/                   # Main package (pip install -e .)
│       ├── __init__.py                    # Logger setup
│       ├── constants/
│       │   └── __init__.py                # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│       ├── entity/
│       │   └── config_entity.py           # @dataclass for each stage
│       ├── utils/
│       │   └── common.py                  # read_yaml, create_dirs, save_json
│       ├── config/
│       │   └── configuration.py           # ConfigurationManager
│       ├── components/                    # CORE ML LOGIC
│       │   ├── data_ingestion.py          # Download, split dataset
│       │   ├── segmentation_validator.py  # Load Roboflow model, validate
│       │   ├── prepare_base_model.py      # EfficientNet setup
│       │   ├── model_trainer.py           # Fine-tune EfficientNet
│       │   └── model_evaluation.py        # Metrics calculation
│       └── pipeline/                      # Orchestrators
│           ├── stage_01_data_ingestion.py
│           ├── stage_02_segmentation_validation.py
│           ├── stage_03_prepare_base_model.py
│           ├── stage_04_model_trainer.py
│           ├── stage_05_model_evaluation.py
│           └── prediction.py              # Inference pipeline
│
├── research/                              # Jupyter prototyping
│   ├── 01_data_ingestion.ipynb
│   ├── 02_segmentation_testing.ipynb
│   ├── 03_prepare_base_model.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
│
├── model/                                 # Deployment models
│   ├── segmentation_model/               # Roboflow exported model
│   │   └── weights/
│   └── regression_model.h5               # Trained EfficientNet
│
├── artifacts/                             # DVC-tracked outputs
│   ├── data_ingestion/
│   │   └── banana_dataset/
│   │       ├── images/                   # Raw photos (7-day span)
│   │       ├── labels/                   # Ground truth CSV
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── segmentation_validation/
│   │   └── validation_report.json        # Roboflow model metrics
│   ├── prepare_base_model/
│   │   ├── base_efficientnet.h5
│   │   └── efficientnet_updated.h5
│   ├── training/
│   │   ├── regression_model.h5
│   │   └── training_history.json
│   └── evaluation/
│       └── metrics.json
│
├── mlruns/                                # MLflow experiment tracking
├── logs/                                  # Auto-generated logs
│
├── templates/
│   └── index.html                         # Flask web interface
│
├── main.py                                # Training entry point
├── app.py                                 # Flask serving API
├── params.yaml                            # ALL hyperparameters
├── dvc.yaml                               # Pipeline definition
├── scores.json                            # Evaluation output (DVC metric)
├── setup.py                               # Package installer
├── requirements.txt                       # Dependencies
├── Dockerfile                             # Container for deployment
└── template.py                            # Project scaffolding script
```

---

## 3. ARCHITECTURE FLOW

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        HOW EVERYTHING CONNECTS                             │
│                                                                            │
│   params.yaml ────────┐                                                    │
│   config/config.yaml ─┼────► ConfigurationManager                          │
│                       │      (reads YAMLs → returns @dataclass objects)    │
│                       │                                                     │
│                       ▼                                                     │
│   ┌────────────────────────────────────────────────────────────┐          │
│   │              PIPELINE STAGES                                │          │
│   │                                                             │          │
│   │  Stage 1: Data Ingestion                                   │          │
│   │    → Download dataset from Google Drive                    │          │
│   │    → Split: 70% train, 15% val, 15% test                   │          │
│   │    → Output: artifacts/data_ingestion/                     │          │
│   │                                                             │          │
│   │  Stage 2: Segmentation Validation (Optional)               │          │
│   │    → Load Roboflow YOLOv8 model                            │          │
│   │    → Run on validation set                                 │          │
│   │    → Calculate mAP@50, mAP@50-95                           │          │
│   │    → Output: validation_report.json                        │          │
│   │                                                             │          │
│   │  Stage 3: Prepare Base Model                               │          │
│   │    → Load EfficientNet-B0 (ImageNet pretrained)            │          │
│   │    → Remove top classification layer                       │          │
│   │    → Add custom regression head                            │          │
│   │    → Output: base_efficientnet.h5                          │          │
│   │                                                             │          │
│   │  Stage 4: Model Training                                   │          │
│   │    → Phase 1: Freeze backbone, train head (5 epochs)       │          │
│   │    → Phase 2: Unfreeze last N layers (params.yaml)         │          │
│   │    → Phase 3: Fine-tune entire model (20+ epochs)          │          │
│   │    → Loss: Huber Loss + Ordinal Penalty                    │          │
│   │    → Output: regression_model.h5                           │          │
│   │                                                             │          │
│   │  Stage 5: Model Evaluation                                 │          │
│   │    → Test set inference                                    │          │
│   │    → Calculate: MAE, RMSE, R², Within-1-day accuracy       │          │
│   │    → MLflow tracking                                       │          │
│   │    → Output: scores.json                                   │          │
│   │                                                             │          │
│   └────────────────────────────────────────────────────────────┘          │
│                       │                                                     │
│                       ▼                                                     │
│   main.py: Runs stages 1→2→3→4→5 sequentially                            │
│   dvc.yaml: DVC pipeline (run with `dvc repro`)                           │
│                       │                                                     │
│                       ▼                                                     │
│   artifacts/ (DVC-tracked)                                                 │
│   model/ (deployment-ready models)                                         │
│   scores.json (metrics)                                                    │
│   mlruns/ (MLflow experiments)                                             │
│                       │                                                     │
│                       ▼                                                     │
│   app.py (Flask) ──► prediction.py ──► model/                             │
│   User uploads image → Segmentation → Regression → Annotated output       │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. PIPELINE STAGES BREAKDOWN

### Stage 1: Data Ingestion

**Purpose**: Download and organize dataset

**Component**: `components/data_ingestion.py`

**Logic**:
```python
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        # Download from Google Drive URL
        # Uses gdown library
        pass
    
    def extract_zip(self):
        # Unzip dataset
        pass
    
    def split_dataset(self):
        # Split into train/val/test based on config ratios
        # Preserve temporal ordering (don't shuffle bananas from same day)
        pass
```

**Inputs**:
- Google Drive URL (config.yaml)
- Split ratios (config.yaml)

**Outputs**:
- `artifacts/data_ingestion/banana_dataset/train/`
- `artifacts/data_ingestion/banana_dataset/val/`
- `artifacts/data_ingestion/banana_dataset/test/`

**Dataset Structure Expected**:
```
banana_dataset/
├── images/
│   ├── banana_01_day_00_front.jpg
│   ├── banana_01_day_00_back.jpg
│   ├── banana_01_day_01_front.jpg
│   └── ...
└── labels.csv  # columns: filename, days_until_death, banana_id, day_number
```

---

### Stage 2: Segmentation Validation (Optional but Recommended)

**Purpose**: Validate your Roboflow segmentation model performance

**Component**: `components/segmentation_validator.py`

**Logic**:
```python
class SegmentationValidator:
    def __init__(self, config: SegmentationValidationConfig):
        self.config = config
        self.model = None
    
    def load_roboflow_model(self):
        # Load YOLOv8-seg model from Roboflow export
        # API key from config
        pass
    
    def run_validation(self):
        # Run on validation images
        # Calculate mAP@50, mAP@50-95
        # Precision, Recall, F1
        pass
    
    def save_metrics(self):
        # Save to validation_report.json
        pass
```

**Why This Stage**:
- Confirms Roboflow model works on your data split
- Provides baseline segmentation metrics
- Useful for debugging if predictions fail

**Outputs**:
- `artifacts/segmentation_validation/validation_report.json`

**Metrics**:
```json
{
  "mAP@50": 0.89,
  "mAP@50-95": 0.76,
  "precision": 0.91,
  "recall": 0.87,
  "f1_score": 0.89,
  "inference_time_ms": 45
}
```

---

### Stage 3: Prepare Base Model

**Purpose**: Set up EfficientNet backbone with custom regression head

**Component**: `components/prepare_base_model.py`

**Logic**:
```python
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_efficientnet(self):
        # Load EfficientNet-B0 from Keras Applications
        # weights='imagenet', include_top=False
        pass
    
    def build_regression_head(self):
        # Add custom layers:
        # GlobalAveragePooling2D
        # Dense(512) + ReLU + BatchNorm + Dropout
        # Dense(256) + ReLU + BatchNorm + Dropout
        # Dense(128) + ReLU
        # Dense(1, activation='linear')  # Days prediction
        pass
    
    def update_base_model(self):
        # Combine backbone + head
        # Freeze backbone initially
        # Save as base_efficientnet_updated.h5
        pass
```

**Inputs**:
- `params.yaml`: IMAGE_SIZE, INCLUDE_TOP, WEIGHTS, CLASSES
- EfficientNet architecture choice (B0/B1/B2)

**Outputs**:
- `artifacts/prepare_base_model/base_efficientnet.h5` (backbone only)
- `artifacts/prepare_base_model/efficientnet_updated.h5` (backbone + head)

**Architecture**:
```
Input: (224, 224, 3)
    ↓
EfficientNet-B0 Backbone (frozen initially)
    ↓
Feature Maps: (7, 7, 1280)
    ↓
GlobalAveragePooling2D → (1280,)
    ↓
Dense(512) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + ReLU
    ↓
Dense(1, linear) → Days prediction
```

---

### Stage 4: Model Training

**Purpose**: Fine-tune EfficientNet for days-to-death regression

**Component**: `components/model_trainer.py`

**Logic**:
```python
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
    
    def load_base_model(self):
        # Load from Stage 3 output
        pass
    
    def prepare_data_generators(self):
        # ImageDataGenerator with augmentation
        # Read labels.csv to get ground truth
        # Create (image, days_left) pairs
        pass
    
    def custom_loss_function(self):
        # Huber Loss (robust to outliers)
        # + Ordinal penalty (wrong direction is worse)
        pass
    
    def train_phase_1_head_only(self):
        # Freeze backbone
        # Train only regression head
        # 5 epochs, lr=1e-3
        pass
    
    def train_phase_2_partial_unfreeze(self):
        # Unfreeze last N layers (from params.yaml)
        # Train 10 epochs, lr=1e-4
        pass
    
    def train_phase_3_full_finetune(self):
        # Unfreeze entire model
        # Train 20-30 epochs, lr=1e-5
        # Early stopping on validation loss
        pass
    
    def save_model(self):
        # Save to artifacts/training/regression_model.h5
        # Copy to model/ for deployment
        pass
```

**Training Schedule**:
```
Phase 1 (Warmup): 5 epochs
  - Freeze: All EfficientNet layers
  - Train: Only regression head
  - LR: 1e-3
  - Optimizer: Adam

Phase 2 (Partial Fine-tune): 10 epochs
  - Freeze: First N layers (params: FREEZE_LAYERS)
  - Train: Last N layers + head
  - LR: 1e-4
  - Optimizer: Adam

Phase 3 (Full Fine-tune): 20-30 epochs
  - Freeze: None
  - Train: All layers
  - LR: 1e-5
  - Optimizer: Adam
  - Early stopping: patience=5
```

**Data Augmentation**:
```python
ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    fill_mode='nearest'
)
```

**Loss Function**:
```python
def combined_loss(y_true, y_pred):
    # Huber loss: Robust to labeling noise
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
    
    # Ordinal penalty: Penalize wrong direction more
    # Predicting 7 when truth is 2 is worse than predicting 4
    direction_error = tf.abs(tf.sign(y_pred - y_true) * tf.square(y_pred - y_true))
    ordinal = tf.reduce_mean(direction_error)
    
    return 0.7 * huber + 0.3 * ordinal
```

**Inputs**:
- `artifacts/prepare_base_model/efficientnet_updated.h5`
- `artifacts/data_ingestion/banana_dataset/train/`
- `artifacts/data_ingestion/banana_dataset/val/`
- `params.yaml`: EPOCHS, BATCH_SIZE, LEARNING_RATE, etc.

**Outputs**:
- `artifacts/training/regression_model.h5`
- `artifacts/training/training_history.json`
- MLflow logged metrics

---

### Stage 5: Model Evaluation

**Purpose**: Calculate performance metrics on test set

**Component**: `components/model_evaluation.py`

**Logic**:
```python
class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
    
    def load_model(self):
        # Load trained model from Stage 4
        pass
    
    def run_inference(self):
        # Predict on test set
        pass
    
    def calculate_metrics(self):
        # MAE (Mean Absolute Error)
        # RMSE (Root Mean Squared Error)
        # R² Score
        # Within-1-day accuracy
        # Within-0.5-day accuracy
        # Direction accuracy (over/under prediction)
        pass
    
    def log_to_mlflow(self):
        # Log metrics, model, artifacts
        pass
    
    def save_metrics(self):
        # Save to scores.json (DVC metric)
        pass
```

**Metrics Calculated**:

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| **MAE** | mean(\|pred - true\|) | < 1.0 day | Average error in days |
| **RMSE** | sqrt(mean((pred - true)²)) | < 1.5 days | Penalizes large errors more |
| **R²** | 1 - (SS_res / SS_tot) | > 0.80 | How well model explains variance |
| **Within-1-day** | % predictions within ±1 day | > 70% | Practical accuracy |
| **Within-0.5-day** | % predictions within ±0.5 day | > 50% | High precision |
| **Direction Acc** | % correct over/under prediction | > 80% | Avoids wrong direction |

**Additional Analysis**:
```python
# Error distribution by ripeness stage
errors_by_stage = {
    "green (7-10 days)": MAE,
    "yellow (4-6 days)": MAE,
    "spotted (1-3 days)": MAE,
    "overripe (0 days)": MAE
}

# Temporal consistency check
# If same banana tracked over multiple days,
# predictions should decrease monotonically
temporal_violations = count_violations()
```

**Outputs**:
- `artifacts/evaluation/metrics.json`
- `scores.json` (DVC metric file)
- MLflow experiment with all metrics

**Example scores.json**:
```json
{
  "test_mae": 0.87,
  "test_rmse": 1.12,
  "test_r2": 0.84,
  "within_1_day_accuracy": 0.73,
  "within_0.5_day_accuracy": 0.51,
  "direction_accuracy": 0.82,
  "inference_time_ms": 23,
  "error_by_stage": {
    "green": 1.2,
    "yellow": 0.8,
    "spotted": 0.6,
    "overripe": 0.3
  }
}
```

---

## 5. CONFIGURATION FILES

### config/config.yaml

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://drive.google.com/file/d/YOUR_DATASET_ID/view?usp=sharing
  local_data_file: artifacts/data_ingestion/banana_dataset.zip
  unzip_dir: artifacts/data_ingestion
  dataset_dir: artifacts/data_ingestion/banana_dataset
  train_dir: artifacts/data_ingestion/banana_dataset/train
  val_dir: artifacts/data_ingestion/banana_dataset/val
  test_dir: artifacts/data_ingestion/banana_dataset/test
  split_ratio:
    train: 0.70
    val: 0.15
    test: 0.15

segmentation_validation:
  root_dir: artifacts/segmentation_validation
  roboflow_api_key: ${ROBOFLOW_API_KEY}  # Environment variable
  roboflow_workspace: your-workspace
  roboflow_project: banana-segmentation
  roboflow_version: 1
  validation_data: artifacts/data_ingestion/banana_dataset/val
  report_path: artifacts/segmentation_validation/validation_report.json

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_efficientnet.h5
  updated_base_model_path: artifacts/prepare_base_model/efficientnet_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/regression_model.h5
  training_data: artifacts/data_ingestion/banana_dataset/train
  validation_data: artifacts/data_ingestion/banana_dataset/val
  history_path: artifacts/training/training_history.json
  mlflow_uri: https://dagshub.com/YOUR_USERNAME/banana-predictor.mlflow

evaluation:
  test_data: artifacts/data_ingestion/banana_dataset/test
  trained_model_path: artifacts/training/regression_model.h5
  metrics_path: artifacts/evaluation/metrics.json
  scores_file: scores.json
  mlflow_uri: https://dagshub.com/YOUR_USERNAME/banana-predictor.mlflow
```

### params.yaml

```yaml
# Data Augmentation
AUGMENTATION: True
AUGMENTATION_ROTATION: 15
AUGMENTATION_BRIGHTNESS: 0.2
AUGMENTATION_ZOOM: 0.1
AUGMENTATION_HORIZONTAL_FLIP: True

# Model Architecture
ARCHITECTURE: EfficientNetB0  # Options: B0, B1, B2
IMAGE_SIZE: [224, 224, 3]
INCLUDE_TOP: False
WEIGHTS: imagenet

# Regression Head
HEAD_UNITS: [512, 256, 128]
DROPOUT_RATE: 0.3
ACTIVATION: relu
FINAL_ACTIVATION: linear  # For regression

# Training - Phase 1 (Head Only)
PHASE_1_EPOCHS: 5
PHASE_1_LR: 0.001
PHASE_1_FREEZE_BACKBONE: True

# Training - Phase 2 (Partial Unfreeze)
PHASE_2_EPOCHS: 10
PHASE_2_LR: 0.0001
PHASE_2_UNFREEZE_LAYERS: 20  # Unfreeze last N layers

# Training - Phase 3 (Full Fine-tune)
PHASE_3_EPOCHS: 30
PHASE_3_LR: 0.00001
PHASE_3_EARLY_STOPPING_PATIENCE: 5

# General Training
BATCH_SIZE: 16
OPTIMIZER: adam
LOSS_FUNCTION: combined  # huber + ordinal penalty
HUBER_DELTA: 1.0
ORDINAL_WEIGHT: 0.3

# Metrics
METRICS:
  - mae
  - mse
  - rmse

# Callbacks
TENSORBOARD: True
MODEL_CHECKPOINT: True
EARLY_STOPPING: True
REDUCE_LR_ON_PLATEAU: True
```

---

## 6. ENTITY DEFINITIONS

**File**: `src/bananaPredictor/entity/config_entity.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    split_ratio: dict

@dataclass(frozen=True)
class SegmentationValidationConfig:
    root_dir: Path
    roboflow_api_key: str
    roboflow_workspace: str
    roboflow_project: str
    roboflow_version: int
    validation_data: Path
    report_path: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_weights: str
    params_include_top: bool
    params_architecture: str
    params_head_units: list
    params_dropout_rate: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    history_path: Path
    mlflow_uri: str
    params_epochs_phase_1: int
    params_epochs_phase_2: int
    params_epochs_phase_3: int
    params_batch_size: int
    params_image_size: list
    params_lr_phase_1: float
    params_lr_phase_2: float
    params_lr_phase_3: float
    params_unfreeze_layers: int
    params_augmentation: bool
    all_params: dict

@dataclass(frozen=True)
class EvaluationConfig:
    test_data: Path
    trained_model_path: Path
    metrics_path: Path
    scores_file: Path
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    all_params: dict
```

---

## 7. COMPONENTS DEEP DIVE

### Component File Structure Pattern

Every component follows this pattern:

```python
# src/bananaPredictor/components/example_component.py

from bananaPredictor.entity.config_entity import ExampleConfig
from bananaPredictor import logger
import os

class ExampleComponent:
    def __init__(self, config: ExampleConfig):
        self.config = config
    
    def method_1(self):
        """First step of the component"""
        logger.info("Starting method_1")
        # Logic here
        logger.info("Completed method_1")
    
    def method_2(self):
        """Second step of the component"""
        logger.info("Starting method_2")
        # Logic here
        logger.info("Completed method_2")
```

### Key Design Principles

1. **Single Responsibility**: Each component does ONE thing
2. **Config-Driven**: All behavior controlled via config dataclass
3. **Logging**: Every major step logged
4. **Error Handling**: Try-except blocks with informative errors
5. **Testability**: Pure functions where possible

---

## 8. DVC PIPELINE

**File**: `dvc.yaml`

```yaml
stages:
  data_ingestion:
    cmd: python src/bananaPredictor/pipeline/stage_01_data_ingestion.py
    deps:
      - src/bananaPredictor/pipeline/stage_01_data_ingestion.py
      - src/bananaPredictor/components/data_ingestion.py
      - config/config.yaml
    outs:
      - artifacts/data_ingestion/banana_dataset

  segmentation_validation:
    cmd: python src/bananaPredictor/pipeline/stage_02_segmentation_validation.py
    deps:
      - src/bananaPredictor/pipeline/stage_02_segmentation_validation.py
      - src/bananaPredictor/components/segmentation_validator.py
      - artifacts/data_ingestion/banana_dataset/val
      - config/config.yaml
    outs:
      - artifacts/segmentation_validation/validation_report.json

  prepare_base_model:
    cmd: python src/bananaPredictor/pipeline/stage_03_prepare_base_model.py
    deps:
      - src/bananaPredictor/pipeline/stage_03_prepare_base_model.py
      - src/bananaPredictor/components/prepare_base_model.py
      - config/config.yaml
    params:
      - IMAGE_SIZE
      - INCLUDE_TOP
      - WEIGHTS
      - ARCHITECTURE
      - HEAD_UNITS
      - DROPOUT_RATE
    outs:
      - artifacts/prepare_base_model/base_efficientnet.h5
      - artifacts/prepare_base_model/efficientnet_updated.h5

  model_training:
    cmd: python src/bananaPredictor/pipeline/stage_04_model_trainer.py
    deps:
      - src/bananaPredictor/pipeline/stage_04_model_trainer.py
      - src/bananaPredictor/components/model_trainer.py
      - artifacts/prepare_base_model/efficientnet_updated.h5
      - artifacts/data_ingestion/banana_dataset/train
      - artifacts/data_ingestion/banana_dataset/val
      - config/config.yaml
    params:
      - PHASE_1_EPOCHS
      - PHASE_2_EPOCHS
      - PHASE_3_EPOCHS
      - BATCH_SIZE
      - PHASE_1_LR
      - PHASE_2_LR
      - PHASE_3_LR
      - PHASE_2_UNFREEZE_LAYERS
      - AUGMENTATION
    outs:
      - artifacts/training/regression_model.h5
      - artifacts/training/training_history.json

  model_evaluation:
    cmd: python src/bananaPredictor/pipeline/stage_05_model_evaluation.py
    deps:
      - src/bananaPredictor/pipeline/stage_05_model_evaluation.py
      - src/bananaPredictor/components/model_evaluation.py
      - artifacts/training/regression_model.h5
      - artifacts/data_ingestion/banana_dataset/test
      - config/config.yaml
    params:
      - IMAGE_SIZE
      - BATCH_SIZE
    metrics:
      - scores.json:
          cache: false
    outs:
      - artifacts/evaluation/metrics.json
```

**Run with**:
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_training

# Visualize pipeline
dvc dag
```

---

## 9. EVALUATION METRICS

### Segmentation Metrics (Stage 2)

**For YOLOv8 Segmentation Model**:

| Metric | Description | Target |
|--------|-------------|--------|
| **mAP@50** | Mean Average Precision at IoU=0.5 | > 0.85 |
| **mAP@50-95** | mAP averaged over IoU 0.5 to 0.95 | > 0.70 |
| **Precision** | TP / (TP + FP) | > 0.90 |
| **Recall** | TP / (TP + FN) | > 0.85 |
| **F1 Score** | 2 * (P * R) / (P + R) | > 0.87 |

**Why these matter**:
- **mAP@50**: Industry standard for object detection
- **mAP@50-95**: More strict, ensures accurate masks
- **Precision**: Avoids false banana detections
- **Recall**: Doesn't miss bananas in image

### Regression Metrics (Stage 5)

**For Days-to-Death Prediction**:

| Metric | Formula | Target | Why It Matters |
|--------|---------|--------|----------------|
| **MAE** | mean(\|pred - actual\|) | < 1.0 day | Average error |
| **RMSE** | sqrt(mean((pred - actual)²)) | < 1.5 days | Penalizes big mistakes |
| **R²** | 1 - SS_res/SS_tot | > 0.80 | Variance explained |
| **Within-1-day** | % within ±1 day | > 70% | Practical accuracy |
| **Direction Acc** | % correct trend | > 80% | Avoids opposite errors |

### Error Analysis Breakdown

**By Ripeness Stage**:
```python
errors_by_stage = {
    "very_green (8-10 days)": {
        "MAE": 1.2,
        "count": 45,
        "note": "Harder to predict - subtle changes"
    },
    "yellow (4-7 days)": {
        "MAE": 0.8,
        "count": 120,
        "note": "Most training data here"
    },
    "spotted (1-3 days)": {
        "MAE": 0.5,
        "count": 80,
        "note": "Easiest - clear visual cues"
    },
    "overripe (0 days)": {
        "MAE": 0.2,
        "count": 35,
        "note": "Obvious state"
    }
}
```

**Confusion Matrix (Bucketed)**:
```
Actual vs Predicted (bucketed into 2-day intervals)

              Predicted
           0-2  2-4  4-6  6-8  8-10
Actual
0-2        40    5    0    0    0
2-4         3   52    8    0    0  
4-6         0    6   48    4    0
6-8         0    0    5   38    2
8-10        0    0    0    3   25
```

### Visualization Metrics

**Plots to Generate**:
1. **Prediction vs Actual Scatter**: Shows correlation
2. **Residual Plot**: Shows bias (over/under prediction)
3. **Error Distribution Histogram**: Shows if errors are symmetric
4. **Predictions Over Time**: For same banana tracked daily
5. **Feature Importance**: Which image regions matter most

---

## 10. DEPLOYMENT STRATEGY

### Inference Pipeline

**File**: `src/bananaPredictor/pipeline/prediction.py`

```python
class BananaPredictionPipeline:
    def __init__(self):
        # Load models once (class-level caching)
        self.segmentation_model = self._load_segmentation_model()
        self.regression_model = self._load_regression_model()
    
    def predict(self, image_path):
        """
        Complete inference pipeline
        
        Returns:
        {
            "total_bananas": 12,
            "predictions": [
                {
                    "banana_id": 1,
                    "bbox": [x1, y1, x2, y2],
                    "days_left": 3.2,
                    "confidence": "high",
                    "category": "ripe"
                },
                ...
            ],
            "summary": {
                "avg_days": 3.8,
                "fresh_count": 3,
                "ripe_count": 8,
                "overripe_count": 1
            }
        }
        """
        # Step 1: Segment bananas
        detections = self.segmentation_model.predict(image_path)
        
        # Step 2: For each banana, predict days
        predictions = []
        for detection in detections:
            crop = self._crop_banana(image_path, detection['bbox'])
            days_left = self.regression_model.predict(crop)
            
            predictions.append({
                "banana_id": detection['id'],
                "bbox": detection['bbox'],
                "days_left": float(days_left),
                "confidence": self._calculate_confidence(detection, days_left),
                "category": self._categorize(days_left)
            })
        
        # Step 3: Aggregate statistics
        summary = self._calculate_summary(predictions)
        
        return {
            "total_bananas": len(predictions),
            "predictions": predictions,
            "summary": summary
        }
```

### Flask API

**File**: `app.py`

```python
from flask import Flask, request, jsonify, render_template
from bananaPredictor.pipeline.prediction import BananaPredictionPipeline
import os

app = Flask(__name__)

# Initialize pipeline once (loads models)
predictor = BananaPredictionPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    
    # Save temporarily
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    
    try:
        # Run prediction
        result = predictor.predict(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Docker Deployment

**File**: `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Copy models
COPY model/ /app/model/

# Expose port
EXPOSE 8080

# Run Flask app
CMD ["python", "app.py"]
```

**Build and Run**:
```bash
# Build image
docker build -t banana-predictor:latest .

# Run container
docker run -p 8080:8080 banana-predictor:latest

# Test
curl -X POST -F "image=@test_banana.jpg" http://localhost:8080/predict
```

---

## 11. INNOVATION POINTS

### Innovation 1: Temporal Consistency Checking

**Problem**: Model might predict 5 days today, 3 days tomorrow (impossible!)

**Solution**: Add temporal smoothing

```python
class TemporalConsistencyChecker:
    def __init__(self):
        self.history = {}  # banana_id → [(day, prediction)]
    
    def check_consistency(self, banana_id, current_day, prediction):
        """
        Enforce: prediction(day_n) < prediction(day_n-1)
        """
        if banana_id in self.history:
            last_day, last_pred = self.history[banana_id][-1]
            
            # Should decrease
            expected_max = last_pred - (current_day - last_day)
            
            if prediction > expected_max:
                # Flag as anomaly
                corrected = expected_max * 0.9  # Conservative correction
                return corrected, "temporal_violation"
        
        self.history[banana_id] = self.history.get(banana_id, [])
        self.history[banana_id].append((current_day, prediction))
        
        return prediction, "valid"
```

### Innovation 2: Uncertainty Estimation

**Problem**: Model gives single point prediction - no confidence

**Solution**: Monte Carlo Dropout

```python
def predict_with_uncertainty(model, image, n_iterations=10):
    """
    Run inference N times with dropout active
    Return: mean prediction ± std deviation
    """
    predictions = []
    
    for _ in range(n_iterations):
        # Enable dropout during inference
        pred = model(image, training=True)
        predictions.append(pred)
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # High std = low confidence
    confidence = "high" if std_pred < 0.5 else "medium" if std_pred < 1.0 else "low"
    
    return {
        "days_left": mean_pred,
        "uncertainty": std_pred,
        "confidence": confidence,
        "confidence_interval": (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
    }
```

### Innovation 3: Attention Visualization

**Problem**: Hard to debug why model made a prediction

**Solution**: Grad-CAM for regression

```python
import tensorflow as tf

def generate_gradcam(model, image, layer_name='top_conv'):
    """
    Generate heatmap showing which regions influenced prediction
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        # For regression, use prediction value directly
        loss = predictions[0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap  # Overlay on original image
```

**Use case**: Show user "Model focused on brown spots in center-right region"

### Innovation 4: Multi-Scale Prediction

**Problem**: Close-up banana vs far-away banana - same image size

**Solution**: Multi-scale input

```python
def multi_scale_predict(model, image):
    """
    Predict at 3 scales, average results
    Helps model generalize to different distances
    """
    scales = [0.8, 1.0, 1.2]
    predictions = []
    
    for scale in scales:
        # Resize image
        h, w = image.shape[:2]
        resized = cv2.resize(image, (int(w*scale), int(h*scale)))
        resized = cv2.resize(resized, (224, 224))  # Back to model size
        
        pred = model.predict(resized)
        predictions.append(pred)
    
    # Weighted average (give more weight to 1.0 scale)
    weights = [0.25, 0.5, 0.25]
    final_pred = np.average(predictions, weights=weights)
    
    return final_pred
```

### Innovation 5: Active Learning for Data Collection

**Problem**: Which new bananas to photograph for maximum model improvement?

**Solution**: Uncertainty sampling

```python
def suggest_data_collection_priority(current_dataset_predictions):
    """
    Analyze current model performance
    Suggest which ripeness stages need more data
    """
    # Group by ripeness stage
    stage_errors = {
        "green": [],
        "yellow": [],
        "spotted": [],
        "overripe": []
    }
    
    for pred in current_dataset_predictions:
        stage = categorize_ripeness(pred['actual_days'])
        error = abs(pred['predicted'] - pred['actual'])
        stage_errors[stage].append(error)
    
    # Calculate average error per stage
    avg_errors = {
        stage: np.mean(errors) if errors else 0
        for stage, errors in stage_errors.items()
    }
    
    # Priority: stages with highest error
    priority = sorted(avg_errors.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "collect_more": priority[0][0],
        "reason": f"High error ({priority[0][1]:.2f} days) in this stage",
        "suggested_count": 20
    }
```

---

## EXECUTION CHECKLIST

### Phase 1: Setup (Week 1)
- [ ] Run `python template.py` to create directory structure
- [ ] Set up virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Configure Google Drive dataset link in `config.yaml`
- [ ] Set up Roboflow API key
- [ ] Initialize DVC (`dvc init`)
- [ ] Set up MLflow tracking (DagsHub/local)

### Phase 2: Data Preparation (Week 2)
- [ ] Run Stage 1: Data Ingestion
- [ ] Verify train/val/test splits
- [ ] Run Stage 2: Segmentation Validation
- [ ] Analyze segmentation metrics
- [ ] Fix any data issues

### Phase 3: Model Development (Week 3-4)
- [ ] Run Stage 3: Prepare Base Model
- [ ] Verify model architecture
- [ ] Run Stage 4: Model Training
  - [ ] Monitor training curves
  - [ ] Adjust hyperparameters if needed
  - [ ] Check for overfitting
- [ ] Run Stage 5: Evaluation
- [ ] Analyze metrics

### Phase 4: Iteration (Week 5)
- [ ] Identify weak points (error analysis)
- [ ] Collect more data if needed
- [ ] Adjust augmentation
- [ ] Re-train with better parameters
- [ ] Achieve target metrics

### Phase 5: Deployment (Week 6)
- [ ] Test prediction pipeline locally
- [ ] Build Flask app
- [ ] Test API endpoints
- [ ] Create Dockerfile
- [ ] Deploy to cloud (AWS/Azure/GCP)

---

## RESUME TALKING POINTS

**When presenting this project:**

> "I built an end-to-end ML system to predict banana shelf life. The pipeline uses instance segmentation (YOLOv8) to isolate individual bananas, then a fine-tuned EfficientNet regressor to predict days until inedible.
>
> Key achievements:
> - **MAE < 1 day** on test set (industry-grade accuracy)
> - **Modular pipeline** with DVC for reproducibility
> - **Custom loss function** combining Huber loss and ordinal penalties
> - **Temporal consistency** checking for multi-day tracking
> - **Deployed** via Flask API with Docker containerization
>
> The challenging part was dataset creation - I photographed banana bunches daily for 7 days, capturing 150+ images with careful labeling. I used retrospective labeling to ensure ground truth accuracy.
>
> This project demonstrates:
> - Computer vision (segmentation + regression)
> - Transfer learning and fine-tuning
> - MLOps practices (DVC, MLflow, CI/CD)
> - Production deployment skills"

---

## FINAL NOTES

This blueprint is **production-grade** but also **learnable**. You can:

1. **Start simple**: Run stages 1, 3, 4, 5 only (skip segmentation validation)
2. **Add complexity gradually**: Add innovations one by one
3. **Scale easily**: Same pattern works for 100+ stages

**Most important**: Follow the pattern religiously. Every new stage follows the EXACT same structure. This makes the codebase predictable, maintainable, and impressive to recruiters.

Good luck! 🍌
