from bananaPredictor.config.configuration import ConfigurationManager
from bananaPredictor.components.model_evaluation import ModelEvaluation
from bananaPredictor import logger


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        model_evaluation = ModelEvaluation(config=eval_config)
        model_evaluation.load_model()
        model_evaluation.run_inference()
        model_evaluation.calculate_metrics()
        model_evaluation.save_metrics()
        model_evaluation.log_to_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
