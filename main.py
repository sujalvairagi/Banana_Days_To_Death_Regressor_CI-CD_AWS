from bananaPredictor import logger
from bananaPredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from bananaPredictor.pipeline.stage_02_seg_data_preparation import SegDataPreparationPipeline
from bananaPredictor.pipeline.stage_03_seg_data_splitter import SegDataSplitterPipeline
from bananaPredictor.pipeline.stage_04_seg_model_trainer import SegModelTrainerPipeline
from bananaPredictor.pipeline.stage_05_segmentation_validation import SegmentationValidationPipeline
from bananaPredictor.pipeline.stage_06_prepare_base_model import PrepareBaseModelTrainingPipeline
from bananaPredictor.pipeline.stage_07_model_trainer import ModelTrainerPipeline
from bananaPredictor.pipeline.stage_08_model_evaluation import ModelEvaluationPipeline


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


# ============ STAGE 2: SEGMENTATION DATA PREPARATION ============
STAGE_NAME = "Segmentation Data Preparation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = SegDataPreparationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 3: SEGMENTATION DATA SPLITTER ============
STAGE_NAME = "Segmentation Data Splitter Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = SegDataSplitterPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 4: SEGMENTATION MODEL TRAINING ============
STAGE_NAME = "Segmentation Model Training Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = SegModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 5: SEGMENTATION VALIDATION ============
STAGE_NAME = "Segmentation Validation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = SegmentationValidationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 6: PREPARE BASE MODEL ============
STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 7: MODEL TRAINING ============
STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ============ STAGE 8: MODEL EVALUATION ============
STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
