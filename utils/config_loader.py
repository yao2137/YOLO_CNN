import yaml
import os

class ConfigLoader:
    @staticmethod
    def load_config(config_path="config/config.yaml"):
        """
        Loads configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config