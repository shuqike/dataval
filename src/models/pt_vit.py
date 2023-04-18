from transformers import ViTImageProcessor, ViTForImageClassification


class ViT:
    def __init__(self) -> None:
        _processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        _model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
