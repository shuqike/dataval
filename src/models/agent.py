import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, Trainer


class Agent:
    def __init__(self, **kwargs) -> None:
        self._seed = kwargs.get('seed', 3407)
        self._max_epoch = kwargs.get('max_epoch', 100)
        self._batch_size = kwargs.get('batch_size', 64)
        self._lr = kwargs.get('lr', 1e-3)
        self._device = kwargs.get('device', 'cpu')
        self._scheduler_name = kwargs.get('scheduler_name', 'linear')
        self._num_warmup_steps = kwargs.get('num_warmup_steps', 0)
        self._metric = evaluate.load(
            kwargs.get('train_metric', 'accuracy')
        )
        self._logging_strategy  = kwargs.get('logging_strategy ', 'epoch')
        self._save_steps = kwargs.get('save_steps', 50)
        self._training_args = TrainingArguments(
            output_dir="test_trainer",
            logging_strategy =self._logging_strategy,
            num_train_epochs=self._max_epoch,
            learning_rate=self._lr,
            lr_scheduler_type=self._scheduler_name,
            warmup_steps=self._num_warmup_steps,
            save_steps=self._save_steps,
            seed=self._seed,
        )
        # Re-initialize weights for data valuation
        pretrained = kwargs.get('pretrained', False)
        if pretrained is False:
            self._model.init_weights()
        # Put model into device
        self._model = self._model.to(self._device)

    def _preproc(self, X):
        return self._preprocessor(X, return_tensors="pt")

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self._metric.compute(
            predictions=predictions,
            references=labels
        )

    def raw_predict(self, X):
        X = self._preproc(X)
        outputs = self._model(**X)
        logits = outputs.logits
        return logits.argmax(-1).item()

    def fit(self, train_dataset, eval_dataset=None):
        '''fit is an offline method
        '''
        trainer = Trainer(
            model=self._model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
