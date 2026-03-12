# Banana Days-to-Death Predictor

Predict how many days until each banana in an image becomes inedible using Instance Segmentation (YOLOv8) and Regression (EfficientNet).

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Run Pipeline

```bash
python main.py
# or
dvc repro
```

## Inference API

```bash
python app.py
```

## Model Setup

### Segmentation Model (YOLOv8 from Roboflow)
1. Go to your trained model on Roboflow
2. Click "Download" → select "YOLOv8" format
3. Place `best.pt` at: `model/segmentation_model/weights/best.pt`

### Regression Model
Trained automatically via the pipeline (Stage 4).
