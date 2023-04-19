from src.models.classifier import Casifier
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification, ConvNextFeatureExtractor, ConvNextForImageClassification


class MobileNet(Casifier):
    def _get_model(self):
        self._model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')

    def _get_processor(self):
        self._processor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')


class ResNet18(Casifier):
    def _get_model(self):
        self._model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')

    def _get_processor(self):
        self._processor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')


class ResNet50(Casifier):
    def _get_model(self):
        self._model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    def _get_processor(self):
        self._processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")


class ConvNeXTTiny(Casifier):
    def _get_model(self):
        self._model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

    def _get_processor(self):
        self._processor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
