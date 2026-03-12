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
        model_eval = ModelEvaluation(config=eval_config)
        model_eval.load_model()
        model_eval.run_inference()
        model_eval.calculate_metrics()
        model_eval.log_to_mlflow()
        model_eval.save_metrics()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
