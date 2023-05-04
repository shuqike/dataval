from time import time
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from src.frameworks.valuator import DynamicValuator
from src.models import RandomForestClassifierDV, RandomForestRegressorDV


class ODVRL(DynamicValuator):
    def __init__(self, problem) -> None:
        """
        Args:
            problem: "clf"
        """
        self.problem = problem
        self.reset()

    def reset(self):
        # create/clear placeholders
        self.data_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list)

    def _calc_oob(self):
        # record starting time
        time_init=time()
        # fit a random forest model
        if self.problem == 'clf': # if it is a classification problem
            self.rf_model=RandomForestClassifierDV(n_estimators=self.n_trees, n_jobs=-1)
        else:
            self.rf_model=RandomForestRegressorDV(n_estimators=self.n_trees, n_jobs=-1)
        self.rf_model.fit(self.X, self.y)
        # record fitting time
        self.time_dict[f'RF_fitting']=time()-time_init
        # record oob value
        self.data_value_dict[f'OOB']=(self.rf_model.evaluate_oob_accuracy(self.X, self.y)).to_numpy()
        # record oob time
        self.time_dict[f'OOB']=time()-time_init
