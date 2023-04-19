from src.models.classifier import Casifier
from transformers import ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, SwinForImageClassification


class ViTbp16(Casifier):
    def _get_model(self):
        return ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def _get_processor(self):
        return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


class SwinTiny(Casifier):
    def _get_model(self):
        return SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def _get_processor(self):
        return AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
