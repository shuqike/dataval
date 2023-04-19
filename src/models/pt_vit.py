from transformers import ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, SwinForImageClassification


class ViTbp16:
    def __init__(self, pretrained=False) -> None:
        # As the Vision Transformer expects each image to be of the same size (resolution),
        # one can use ViTImageProcessor to resize (or rescale) and normalize images for the model.
        self._processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        # Load from pretrained vit model
        self._model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass

    def predict(self, images):
        inputs = self._processor(images=images, return_tensors="pt")
        outputs = self._model(**inputs)
        logits = outputs.logits
        return logits.argmax(-1).item()


class SwinTiny:
    def __init__(self, pretrained=False) -> None:
        self._feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self._model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        # Re-initialize weights for data valuation
        if pretrained is False:
            self._model.init_weights()

    def fit(self, images, labels):
        pass

    def predict(self, images):
        inputs = self._feature_extractor(images=images, return_tensors="pt")
        outputs = self._model(**inputs)
        logits = outputs.logits
        return logits.argmax(-1).item()
