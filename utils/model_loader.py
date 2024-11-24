import os
import torch
import torchreid

class ModelLoader:
    @staticmethod
    def load_yolov5(config):
        """
        Loads the YOLOv5 model based on configuration.

        Args:
            config (dict): Configuration dictionary for YOLOv5.

        Returns:
            YOLOv5 model: Loaded YOLOv5 model.
        """
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv5 model not found at: {model_path}")
        print(f"Loading YOLOv5 model from: {model_path}")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, source="local")
        return model, config["confidence_threshold"]

    @staticmethod
    def load_osnet(config):
        """
        Loads the OSNet model based on configuration.

        Args:
            config (dict): Configuration dictionary for OSNet.

        Returns:
            OSNet model: Loaded OSNet model ready for feature extraction.
        """
        model_path = config["model_path"]
        use_gpu = config["use_gpu"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OSNet model not found at: {model_path}")
        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Loading OSNet model from: {model_path} on {device}")
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model

    @staticmethod
    def load_behavior_classifier(model_path="models/behavior/behavior_classifier.pth", num_classes=5):
        """
        Loads the behavior classification model.
        """
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

