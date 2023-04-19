from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax


def preprocess(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class Tweet_roBERTa_base:
    def __init__(self, pretrained=False, task='sentiment') -> None:
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def fit(self, texts, labels):
        pass
    
    def predict(self, texts):
        encoded_inputs = [
            self._tokenizer(
                preprocess(text),
                return_tensors='pt'
            )
            for text in texts
        ]
        outputs = self._model(**encoded_inputs)
