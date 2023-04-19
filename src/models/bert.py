from agent import Agent
from transformers import AutoModelForSequenceClassification


class Lancer(Agent):
    def __init__(self, **kwargs) -> None:
        num_labels = kwargs.get('num_labels')
        model_family = kwargs.get('model_family', 'bert-base-uncased')
        self._model = AutoModelForSequenceClassification.from_pretrained(model_family, num_labels=num_labels)
        super().__init__(**kwargs)

    def raw_predict(self, X):
        raise NotImplementedError
