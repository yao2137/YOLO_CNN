import logging
import yaml

class Logger:
    @staticmethod
    def setup_logger(config_path="config/logger_config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.basicConfig(**config)
        return logging.getLogger("Main")