from bananaPredictor import logger
from bananaPredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from bananaPredictor.pipeline.stage_02_segmentation_validation import SegmentationValidationPipeline
from bananaPredictor.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from bananaPredictor.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from bananaPredictor.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


# ============ STAGE 1: DATA INGESTION ============
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 2: SEGMENTATION VALIDATION ============
STAGE_NAME = "Segmentation Validation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = SegmentationValidationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 3: PREPARE BASE MODEL ============
STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 4: MODEL TRAINING ============
STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 5: MODEL EVALUATION ============
STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
