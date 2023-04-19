import numpy as np
import torch
import src.utils as utils


class TruncatedMC:
    def __init__(self, train_dataset, test_dataset,
                 model_family, seed=None,
                 perf_metric='accuracy',
                 **kwargs) -> None:
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
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_family = model_family
        self.model = utils.return_model(self.model_family, **kwargs)
        self.perf_metric = perf_metric

        # Sanity check for single/multiclass label
        if len(self.test_dataset) > 2:
            assert self.perf_metric != 'f1', 'Invalid metric for multiclass!'
            assert self.perf_metric != 'auc', 'Invalid metric for multiclass!'

        # Get the score of the initial model
        self.random_score = self._init_score()

    def _init_score(self):
        '''
        Get the score of the initial model
        '''
        if self.perf_metric == 'accuracy':
            hist = np.bincount(self.test_dataset['label']).astype(float) / len(self.test_dataset['label'])
            return np.max(hist)
        else:
            raise NotImplementedError
