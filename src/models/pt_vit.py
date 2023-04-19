from src.models.classifier import Casifier
from transformers import ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, SwinForImageClassification


class ViTbp16(Casifier):
    def __init__(self, **kwargs) -> None:
        # As the Vision Transformer expects each image to be of the same size (resolution),
        # one can use ViTImageProcessor to resize (or rescale) and normalize images for the model.
        self._processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        # Load from pretrained vit model
        self._model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        super().__init__(**kwargs)


class SwinTiny(Casifier):
    def __init__(self, **kwargs) -> None:
        self._processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self._model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        super().__init__(**kwargs)
