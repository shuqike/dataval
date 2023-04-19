from src.models.classifier import Casifier
from transformers import ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, SwinForImageClassification


class ViTbp16(Casifier):
    def _get_model(self):
        self._model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def _get_processor(self):
        self._processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


class SwinTiny(Casifier):
    def _get_model(self):
        self._model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def _get_processor(self):
        self._processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
