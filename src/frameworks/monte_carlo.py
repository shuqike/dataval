import numpy as np
import torch
import utils


class TruncatedMC:
    def __init__(self, X, y, X_test, y_test, sources=None, sample_weight=None,
                 model_family='logistic', metric='accuracy', seed=None,
                 directory=None, **kwargs) -> None:
        '''
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations
            directory: Directory to save results and figures
            **kwargs: Arguments of the model
        '''
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.model_family = model_family
        self.model = utils.return_model(self.model_family, **kwargs)
        self.metric = metric
        self.directory = directory
        # Sanity check for single/multiclass label
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        # Get the score of the initial model
        self.random_score = self._init_score(self.metric)

    def _init_score(self, metric):
        '''
        Get the score of the initial model
        '''
        if metric == 'accuracy':
            return self.model.score(self.X, self.y)
        elif metric == 'f1':
            return self.model.f1_score(self.X, self.y)
        elif metric == 'auc':
            return self.model.auc_score(self.X, self.y)
