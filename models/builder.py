from .cnn import SpeciesClassifier
from .vit_wrapper import ViTWithAttention


def get_model(model_type: str,
              num_classes: int,
              input_channels: int = 4,
              pretrained: bool = True):
    """
    Factory for model instantiation.
    """

    model_type = model_type.lower()

    if model_type == "cnn":
        return SpeciesClassifier(num_classes=num_classes,
                                 input_channels=input_channels)

    elif model_type == "vit":
        return ViTWithAttention(num_classes=num_classes,
                                pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
      
