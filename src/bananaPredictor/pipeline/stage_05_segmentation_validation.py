from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.segmentation_validator import SegmentationValidator
from bananaPredictor import logger


STAGE_NAME = "Segmentation Validation Stage"


class SegmentationValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        seg_val_config = config.get_segmentation_validation_config()
        seg_validator = SegmentationValidator(config=seg_val_config)
        seg_validator.load_model()
        seg_validator.run_validation()
        seg_validator.save_metrics()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SegmentationValidationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
