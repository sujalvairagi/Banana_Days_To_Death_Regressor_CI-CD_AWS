from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.seg_data_preparation import SegDataPreparation
from bananaPredictor import logger


STAGE_NAME = "Segmentation Data Preparation Stage"


class SegDataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        seg_data_config = config.get_seg_data_preparation_config()
        seg_data_prep = SegDataPreparation(config=seg_data_config)
        seg_data_prep.validate_coco_dataset()
        seg_data_prep.convert_coco_to_yolo()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SegDataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
