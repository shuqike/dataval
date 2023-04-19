from src.models.classifier import Casifier
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification, ConvNextFeatureExtractor, ConvNextForImageClassification


class MobileNet(Casifier):
    def __init__(self, **kwargs) -> None:
        self._preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
        self._model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
        super().__init__(**kwargs)


class ResNet18(Casifier):
    def __init__(self, **kwargs) -> None:
        self._processor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')
        self._model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')
        super().__init__(**kwargs)


class ResNet50(Casifier):
    def __init__(self, **kwargs) -> None:
        self._processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self._model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        super().__init__(**kwargs)


class ConvNeXTTiny(Casifier):
    def __init__(self, **kwargs) -> None:
        self._processor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
        self._model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
        super().__init__(**kwargs)
