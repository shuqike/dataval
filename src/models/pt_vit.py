from transformers import ViTImageProcessor, ViTForImageClassification


class ViT:
    def __init__(self) -> None:
        # As the Vision Transformer expects each image to be of the same size (resolution),
        # one can use ViTImageProcessor to resize (or rescale) and normalize images for the model.
        self._processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        # Load from pretrained vit model
        self._model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # Re-initialize weights for data valuation
        self._model.init_weights()

    def fit(self):
        pass
