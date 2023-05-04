import copy
from time import time
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import torch
from src.frameworks.valuator import DynamicValuator
from src.models import RandomForestClassifierDV, RandomForestRegressorDV, Vestimator


class DvrlLoss(torch.nn.Module):
    def __init__(self, epsilon, threshold):
        """Construct class
        :param epsilon: Used to avoid log(0)
        :param threshold: The exploration rate
        """
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, est_data_value, s_input, reward_input):
        """Calculate the loss.
        :param est_data_value: The estimated data value(probability)
        :param s_input: Final selection
        :param reward_input: Reward
        """
        # Generator loss (REINFORCE algorithm)
        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        # TODO: incorporate OOB value
        prob = torch.sum(s_input * torch.log(est_data_value + self.epsilon) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.epsilon))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        dve_loss = (-reward_input * prob) + \
                   1e3 * torch.maximum(torch.mean(est_data_value) - self.threshold, zero) + \
                   1e3 * torch.maximum(1 - self.threshold - torch.mean(est_data_value), zero)

        return dve_loss


class Odvrl(DynamicValuator):
    """Online Data Valuation using Reinforcement Learning (DVRL) class.
    """
    def __init__(self, problem, n_trees, pred_model, parameters) -> None:
        """
        Args:
            problem: "clf"
            n_trees: B, number of weak learner models
        """
        self.problem = problem
        self.n_trees=n_trees
        self.reset()

        # Basic RL parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.outer_iterations = parameters.epoches

        self.device = parameters.device
        # Network parameters for data value estimator
        self.input_dim = parameters.input_dim
        self.hidden_dim = parameters.hidden_dim
        self.output_dim = parameters.output_dim
        self.outer_iterations = parameters.epoches
        self.layer_number = parameters.layer_number
        self.batch_size = parameters.batch_size
        self.learning_rate = parameters.learning_rate

        # prepare the prediction model
        self.pred_model = copy.deepcopy(pred_model)
        if parameters.cuda:
            self.pred_model.cuda()

    def reset(self):
        """create/clear placeholders
        """
        self.data_value_dict=defaultdict(list)
        self.time_dict=defaultdict(list)

    def _calc_oob(self, X, y, X_val=None, y_val=None):
        """Calculate OOB value of the current dataset
        """
        # record starting time
        time_init=time()
        # fit a random forest model
        if self.problem == 'clf': # if it is a classification problem
            self.rf_model=RandomForestClassifierDV(n_estimators=self.n_trees, n_jobs=-1)
        else:
            self.rf_model=RandomForestRegressorDV(n_estimators=self.n_trees, n_jobs=-1)
        self.rf_model.fit(X, y)
        # record fitting time
        self.time_dict[f'RF_fitting']=time()-time_init
        # record oob value
        self.data_value_dict[f'OOB']=(self.rf_model.evaluate_oob_accuracy(X, y)).to_numpy()
        # record oob time
        self.time_dict[f'OOB']=time()-time_init
        # record mean accuracy
        if X_val is not None:
            self.data_value_dict[f'base_metric']=self.rf_model.evaluate_mean_metrics(X_val=X_val, y_val=y_val)

    def one_step(self, X, y, X_val, y_val):
        """Train value estimator, estimate OOB values
        """
        # selection network
        self.value_estimator = Vestimator(
            input_dim=self.input_dim, 
            layer_number=self.layer_number, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        )

        # first OOB with baseline performance
        self._calc_oob(X, y, X_val, y_val)
        print('First round weak learner performance F1: %f' % self.data_value_dict[f'base_metric'])

        self.value_estimator = self.value_estimator.to(self.device)
        # loss function
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold)
        # optimizer
        dvrl_optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.learning_rate)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(dvrl_optimizer, gamma=0.999)

        best_reward = 0
        for epoch in range(self.outer_iterations):
            # change learning rate
            scheduler.step()
            # clean up grads
            self.value_estimator.train()
            self.value_estimator.zero_grad()
            dvrl_optimizer.zero_grad()
