import os
import _pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import get_scheduler
from src.frameworks.valuator import StaticValuator
import src.utils as utils


class TruncatedMC(StaticValuator):
    def __init__(self,
                 model_family,
                 train_dataset,
                 X_train,
                 X_test,
                 test_dataset,
                 sources=None,
                 seed=None,
                 perf_metric='accuracy',
                 directory='../logs',
                 **kwargs) -> None:
        """
        Args:
            model_family: The model family used for learning algorithm
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations
            perf_metric: Evaluation metric
            directory: Directory to save results and figures
            **kwargs: Arguments of the model
        """

        # Sanity check for datasets
        if len(test_dataset) > 2:
            assert perf_metric != 'f1', 'Invalid metric for multiclass!'
            assert perf_metric != 'auc', 'Invalid metric for multiclass!'

        self.directory = directory

        # Record parameters
        self.kwargs = kwargs

        # Prepare datasets
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.X_train = X_train
        self.X_test = X_test
        if sources is None:
            # Each datum has its own source
            sources = {i:np.array([i]) for i in range(len(self.X_train))}
        elif not isinstance(sources, dict):
            # Set down the given sources as a dict
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        self.sources = sources
        self.train_dataset = train_dataset
        self.num_data = len(self.X_train)
        self.test_dataset = test_dataset

        # Prepare model
        self.model_family = model_family
        self.model = utils.return_model(self.model_family, **kwargs)
        self.perf_metric = perf_metric

        # Prepare valuation hyper-parameters
        self.g_shap_lr = kwargs.get('g_shap_lr', None)

        # Get the score of the initial model
        self.random_score = self._init_score()

    def _init_score(self):
        """Get the score of the initial model.
        """
        if self.perf_metric == 'accuracy':
            hist = np.bincount(self.test_dataset['label']).astype(float) / len(self.test_dataset['label'])
            return np.max(hist)
        else:
            raise NotImplementedError

    def _calc_loo(self):
        """Calculate leave-one-out values for the given metric.
        """
        self.model.reset()
        self.model.fit(self.train_dataset, self.test_dataset)
        baseline_val = self.model.perf_metric(self.test_dataset)
        self.vals_loo = np.zeros(self.num_data)
        for i in tqdm(self.sources.keys()):
            X_batch = np.delete(self.X_train, self.sources[i], axis=0)
            y_batch = np.delete(self.train_dataset['label'], self.sources[i], axis=0)
            self.model.reset()
            self.model.fit(
                train_dataset=Dataset.from_dict(
                    {'feature': X_batch, 'label': y_batch}
                )
            )
            removed_val = self.model.perf_metric(self.test_dataset)
            self.vals_loo[self.sources[i]] = (baseline_val - removed_val)
            self.vals_loo[self.sources[i]] /= len(self.sources[i])

    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.model.reset()
        for _ in range(1):
            self.model.fit(self.X_train, self.train_dataset['label'])
            for __ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(
                    self.model.perf_metric(
                        Dataset.from_dict(
                           {'feature': self.X_test[bag_idxs], 'label': self.test_dataset['label'][bag_idxs]}
                        )
                    )
                )
        self.mean_score = np.mean(scores)

    def _tmc_iter(self, tol):
        """Iterate once for tmc-shapley algorithm
        """
        idxs = np.random.permutation(len(self.sources))
        marginal_contribs = np.zeros(len(self.X_train))
        X_batch = np.zeros((0,) + tuple(self.X_train.shape[1:]))
        y_batch = np.zeros(0, int)
        truncation_counter = 0
        new_score = self.random_score
        for idx in tqdm(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X_train[self.sources[idx]]])
            y_batch = np.concatenate([y_batch, self.train_dataset['label'][self.sources[idx]]])
            if len(set(y_batch)) == len(set(self.test_dataset['label'])):
                self.model.reset()
                self.model.fit(
                    train_dataset=Dataset.from_dict(
                        {'feature': X_batch, 'label': y_batch}
                    )
                )
                new_score = self.model.perf_metric(self.test_dataset)
            marginal_contribs[self.sources[idx]] = (new_score - old_score)
            marginal_contribs[self.sources[idx]] /= len(self.sources[idx])
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tol * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def _calc_tmcshap(self, num_iter, tol):
        """Runs TMC-Shapley algorithm.
        Args:
            num_iter: Number of iterations to run.
            tol: Truncation tolerance ratio.
        """
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        marginals, idxs = [], []
        for _ in range(num_iter):
            marginals, idxs = self._tmc_iter(tol)
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])
        self.vals_tmcshap = np.mean(self.mem_tmc, axis=0)

    def _get_gshap_lr(self):
        return 1e-3

    def _gshap_iter(self, perm_idxs):
        """Iterate once for gradient-shapley algorithm
        """
        val_result = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.g_shap_lr)
        for i in tqdm(perm_idxs):
            self.model._model.train()
            outputs = self.model._model(self.X_train[self.sources[i]], self.train_dataset['label'][self.sources[i]])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.model._model.eval()
            val_result.append(self.model.perf_metric(self.test_dataset))
        return np.asarray(val_result)

    def _calc_gshap(self, num_iter):
        """Method for running G-Shapley algorithm.
        Args:
            num_iter: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None calculates the best learning rate.
        """
        if self.g_shap_lr is None:
            self.g_shap_lr = self._get_gshap_lr()
        for _ in range(num_iter):
            # initialize parameters
            self.model.reset()
            # initialize marginal contributions
            marginal_contribs = np.zeros(len(self.sources.keys()))
            # random permutation of train data points
            perm_idxs = list(self.sources.keys())
            np.random.shuffle(perm_idxs)
            # Get valuation at each training epoch
            val_result = self._gshap_iter(perm_idxs)
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.X_train))
            for i, idx in enumerate(perm_idxs):
                individual_contribs[self.sources[idx]] += marginal_contribs[i]
                individual_contribs[self.sources[idx]] /= len(self.sources[idx])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))]
            )
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(perm_idxs, (1,-1))]
            )
        self.vals_gshap = np.mean(self.mem_gshap, axis=0)

    def _save_results(self, overwrite=False):
        """Saves results computed so far.
        """
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory, 
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )  
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))  

    def run(self, save_every, err, tol=1e-2, do_tmc = True, do_gshap=False, do_loo=False):
        """
        Calculates datum values
        Args:
            save_every: save marginal contributions every n iterations
            err: stopping criteria
            tol: truncation tolerance. If None, it's computed
            gshap: whether to use G-Shapley to compute marginal contributions
            loo: whether to use Leave-One-Out to compute marginal contributions
        """
        if do_loo:
            if hasattr(self, 'vals_loo') == False:
                tqdm.write('Calculate leave-one-out')
                self._calc_loo()

        self.mem_tmc = np.zeros((0, self.num_data))
        self.mem_gshap = np.zeros((0, self.num_data))
        while do_tmc or do_gshap:
            if do_gshap:
                if utils.error(self.mem_gshap) < err:
                    do_gshap = False
                else:
                    self._calc_gshap(save_every, tol)
            if do_tmc:
                if utils.error(self.mem_tmc) < err:
                    do_tmc = False
                else:
                    self._calc_tmcshap(save_every, tol)
            self._save_results()
