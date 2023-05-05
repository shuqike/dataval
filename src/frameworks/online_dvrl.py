import os
import copy
from time import time
import warnings
warnings.filterwarnings("ignore")
import tqdm
import numpy as np
import torch
import src.utils as utils
from src.frameworks.valuator import DynamicValuator
from src.models import Vestimator


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
    def __init__(self, num_weak, pred_model, val_model, value_estimator, parameters) -> None:
        """
        Args:
            num_weak: B, number of weak learner models
        """
        self.num_weak=num_weak

        self.saving_path = parameters.saving_path  # '../logs/odvrl/selection_network'

        # Basic RL parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.val_batch_size = parameters.val_batch_size  # Batch size for validation 
        self.outer_iterations = parameters.epochs

        self.device = parameters.device
        self.num_workers = parameters.num_workers
        # Network parameters for data value estimator
        # self.input_dim = parameters.input_dim
        # self.hidden_dim = parameters.hidden_dim
        # self.output_dim = parameters.output_dim
        # self.layer_number = parameters.layer_number
        self.learning_rate = parameters.learning_rate

        # prepare the original model, prediction model and validation model
        self.ori_model = copy.deepcopy(pred_model)
        self.pred_model = copy.deepcopy(pred_model)
        self.val_model = val_model
        self.val_model.to(self.device)

        # selection network
        self.value_estimator = value_estimator
        # self.value_estimator = Vestimator(
        #     input_dim=self.input_dim, 
        #     layer_num=self.layer_number, 
        #     hidden_num=self.hidden_dim, 
        #     output_dim=self.output_dim
        # )

    def _test_acc(self, model, val_dataset):
        pred_list = []
        label_list = []
        model.eval()
        model.to(self.device)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=True)
        for batch_data in val_loader:
            feature, label = batch_data

            feature = feature.to(self.device)
            label = label.to(self.device)

            output = model(feature)
            _, pred = torch.max(output, 1)

            pred = pred.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            pred_list += pred.tolist()
            label_list += label.tolist()

        valid_perf = sum(1 for x, y in zip(pred_list, label_list) if x == y) / len(pred_list)  # accuracy
        return valid_perf

    def evaluate(self, X, y):
        self.val_model.eval()
        X = torch.unsqueeze(X, 1)
        X = X.to(self.device)
        y = y.to(self.device)
        label_one_hot = torch.nn.functional.one_hot(y).to(self.device)
        label_val_pred = self.val_model(X.float())
        y_train_hat = torch.abs(label_one_hot - label_val_pred)
        values =  self.value_estimator(X, label_one_hot, y_train_hat)
        values = torch.squeeze(values)
        return values.cpu().detach().numpy()

    def one_step(self, step_id, X, y, val_dataset):
        """Train value estimator, estimate OOB values
        """
        # first OOB with baseline performance
        # baseline performance
        valid_perf = self._test_acc(self.ori_model, val_dataset)

        # load evaluation network to device
        self.value_estimator = self.value_estimator.to(self.device)
        # loss function
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold).to(self.device)
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
            
            # train predictor from scratch everytime
            new_model = copy.copy(self.pred_model)
            new_model.to(self.device)
            # predictor optimizer
            pre_optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)

            # use a list save all s_input and data values
            data_value_list = []
            s_input = []

            for batch_si in (16, 32):
                train_loader = utils.CustomDataloader(X=X, y=y, batch_size=int(batch_si))
                for batch_data in train_loader:
                    self.val_model.eval()
                    new_model.train()
                    new_model.zero_grad()
                    pre_optimizer.zero_grad()

                    feature, label = batch_data
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                    label_one_hot = torch.nn.functional.one_hot(label, num_classes=10)
                    label_val_pred = self.val_model(feature.float())

                    y_pred_diff = torch.abs(label_one_hot - label_val_pred)

                    # selection estimation
                    est_dv_curr = self.value_estimator(feature, label_one_hot, y_pred_diff)
                    data_value_list.append(est_dv_curr)

                    # 'sel_prob_curr' is a 01 vector generated by binomial distributions
                    sel_prob_curr = np.random.binomial(1, est_dv_curr.cpu().detach().numpy(), est_dv_curr.shape)
                    selected_idxs = np.where(sel_prob_curr)[0]
                    sel_prob_curr = torch.Tensor(sel_prob_curr).to(self.device)
                    s_input.append(sel_prob_curr)

                    # train new model
                    output = new_model(feature[selected_idxs].float())
                    # loss function
                    pre_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = pre_criterion(output, label[selected_idxs])

                    # back propagation
                    loss.mean().backward()
                    pre_optimizer.step()

                    del batch_data, feature, label, label_one_hot, label_val_pred, y_pred_diff, output
                    utils.super_save()

                del train_loader
                utils.super_save()

            # test the performance of the new model
            dvrl_perf = self._test_acc(new_model, val_dataset)
            reward = dvrl_perf - valid_perf

            if reward > best_reward:
                best_reward = reward
                flag_save = True
            else:
                flag_save = False

            # update the selection network
            reward = torch.Tensor([reward])
            data_value_list = torch.cat(data_value_list, 0)
            s_input = torch.cat(s_input, 0)
            reward = reward.to(self.device)
            data_value_list = data_value_list.to(self.device)
            s_input = s_input.to(self.device)
            loss = dvrl_criterion(data_value_list, s_input, reward)
            print(
                'At step %d epoch %d, the reward is %f, the prob is %f' % (step_id, epoch, \
                reward.cpu().detach().numpy()[0], \
                np.max(data_value_list.cpu().detach().numpy()))
            )
            loss.backward()
            dvrl_optimizer.step()

            del new_model
            utils.super_save()

            if flag_save or epoch % 50 ==0:
                torch.save(
                    self.value_estimator.state_dict(), 
                    os.path.join(
                        self.saving_path, 
                        'net_epoch%d.pth' % (epoch + 1)
                    )
                )
