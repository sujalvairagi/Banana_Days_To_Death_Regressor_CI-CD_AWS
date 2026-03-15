from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.seg_data_splitter import SegDataSplitter
from bananaPredictor import logger


STAGE_NAME = "Segmentation Data Splitter Stage"


class SegDataSplitterPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        seg_split_config = config.get_seg_data_splitter_config()
        seg_splitter = SegDataSplitter(config=seg_split_config)
        seg_splitter.split_dataset()
        seg_splitter.create_dataset_yaml()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SegDataSplitterPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
