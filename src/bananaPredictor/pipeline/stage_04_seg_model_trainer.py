from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.seg_model_trainer import SegModelTrainer
from bananaPredictor import logger


STAGE_NAME = "Segmentation Model Training Stage"


class SegModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        seg_train_config = config.get_seg_model_training_config()
        seg_trainer = SegModelTrainer(config=seg_train_config)
        seg_trainer.train()
        seg_trainer.export_best_weights()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = SegModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
