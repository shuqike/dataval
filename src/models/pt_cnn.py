import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification, ConvNextFeatureExtractor, ConvNextForImageClassification


class MobileNet:
    def __init__(self, pretrained=False) -> None:
        self._preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
        self._model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass

    def predict(self, images):
        inputs = self._preprocessor(images, return_tensors="pt")
        outputs = self._model(**inputs)
        logits = outputs.logits
        return logits.argmax(-1).item()


class ResNet18:
    def __init__(self, pretrained=False) -> None:
        self._feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')
        self._model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass

    def predict(self, images):
        inputs = self._feature_extractor(images, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return logits.argmax(-1).item()


class ResNet50:
    def __init__(self, pretrained=False) -> None:
        self._processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self._model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass

    def predict(self, images):
        inputs = self._processor(images, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return logits.argmax(-1).item()


class ConvNeXTTiny:
    def __init__(self, pretrained=False) -> None:
        self._feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
        self._model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass
    
    def predict(self, images):
        inputs = self._feature_extractor(images, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        return logits.argmax(-1).item()
