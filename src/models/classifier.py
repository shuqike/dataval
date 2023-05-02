import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification


class Casifier:
    """General classifier class
    """
    def _get_model(self): raise NotImplementedError
    def _get_processor(self): self._processor = lambda X, **kwargs : X

    def __init__(self, **kwargs) -> None:
        self._get_model()
        self._get_processor()
        self._seed = kwargs.get('seed', 3407)
        # TODO: DEBUG
        self._max_epoch = kwargs.get('max_epoch', 2)
        self._batch_size = kwargs.get('batch_size', 16)
        self._lr = kwargs.get('lr', 1e-3)
        self._device = kwargs.get('device', 'cpu')
        self._scheduler_name = kwargs.get('scheduler_name', 'linear')
        self._num_warmup_steps = kwargs.get('num_warmup_steps', 0)
        # Train metric for recording training results
        self._train_metric = evaluate.load(
            kwargs.get('train_metric', 'accuracy')
        )
        # Performance metric for data valuation
        self._perf_metric = evaluate.load(
            kwargs.get('perf_metric', 'accuracy')
        )
        self._logging_strategy  = kwargs.get('logging_strategy ', 'no')
        self._save_steps = kwargs.get('save_steps', 10)
        self._training_args = TrainingArguments(
            output_dir="test_trainer",
            logging_strategy =self._logging_strategy,
            num_train_epochs=self._max_epoch,
            learning_rate=self._lr,
            lr_scheduler_type=self._scheduler_name,
            warmup_steps=self._num_warmup_steps,
            save_steps=self._save_steps,
            seed=self._seed,
            disable_tqdm=True,
        )
        # Re-initialize weights for data valuation
        self._pretrained = kwargs.get('pretrained', False)
        if self._pretrained  is False:
            self._model.init_weights()
        # Put model into device
        self._model = self._model.to(self._device)
        # Prepare a dummy trainer
        self._trainer = Trainer(model=self._model)

    def _preproc(self, X):
        return self._processor(X, return_tensors="pt")

    def _compute_train_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self._train_metric.compute(
            predictions=predictions,
            references=labels
        )

    def get_device(self):
        return self._device

    def reset(self):
        self._get_model()
        # Re-initialize weights for data valuation
        if self._pretrained is False:
            self._model.init_weights()
        # Put model into device
        self._model = self._model.to(self._device)

    def raw_predict(self, X):
        X = self._preproc(X)
        outputs = self._model(**X)
        logits = outputs.logits
        return logits.argmax(-1).item()

    def predict(self, x):
        return self._model(x)

    def perf_metric(self, eval_dataset):
        eval_results = self._trainer.evaluate(eval_dataset)
        return eval_results['eval_accuracy']

    def one_epoch(self, train_dataset):
        training_args = TrainingArguments(
            output_dir="test_trainer",
            logging_strategy='no',
            num_train_epochs=1,
            learning_rate=self._lr,
            lr_scheduler_type='constant',
            warmup_steps=0,
            save_strategy='no',
            seed=self._seed,
            disable_tqdm=True,
        )
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=self._compute_train_metrics,
        )
        self._trainer.train()

    def fit(self, train_dataset, eval_dataset=None, training_args=None):
        """'fit' is an offline method.
        """
        if training_args is None:
            self._trainer = Trainer(
                model=self._model,
                args=self._training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_train_metrics,
            )
        else:
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_train_metrics,
            )
        self._trainer.train()


class Lancer(Casifier):
    """Language models for classification
    """
    def _get_model(self):
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_family, num_labels=self._num_labels)

    def __init__(self, **kwargs) -> None:
        self._num_labels = kwargs.get('num_labels')
        self._model_family = kwargs.get('model_family', 'bert-base-uncased')
        super().__init__(**kwargs)
