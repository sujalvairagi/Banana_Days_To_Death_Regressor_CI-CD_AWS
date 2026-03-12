from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.model_trainer import ModelTrainer
from bananaPredictor import logger


STAGE_NAME = "Model Training Stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_trainer = ModelTrainer(config=training_config)
        model_trainer.load_base_model()
        model_trainer.prepare_data_generators()
        model_trainer.train()
        model_trainer.save_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
