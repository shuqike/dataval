import numpy as np
from tqdm import tqdm
from datasets import Dataset


class Valuator:
    def _calc_loo(self):
        raise NotImplementedError


class StaticValuator(Valuator):
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


class DynamicValuator(Valuator):
    def _calc_loo(self):
        raise NotImplementedError
