import os
import copy
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sklearn
import tensorflow as tf
import torch
import wandb
import src.utils as utils
from src.frameworks.valuator import DynamicValuator
from src.models import RandomForestClassifierDV


class RlLoss(torch.nn.Module):
    def __init__(self, epsilon):
        """Construct class
        :param epsilon: Used to avoid log(0)
        :param threshold: The exploration rate
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, est_data_value, s_input, reward_input):
        """Calculate the loss.
        :param est_data_value: The estimated data value(probability)
        :param s_input: Final selection
        :param reward_input: Reward
        """
        # Generator loss (REINFORCE algorithm)
        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.epsilon) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.epsilon))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        dve_loss = (-reward_input * prob) + \
                   1e3 * torch.mean(est_data_value) + \
                   1e3 * (1 - torch.mean(est_data_value))

        return dve_loss


class Frequp(DynamicValuator):
    """Online Data Valuation using Reinforcement Learning (DVRL) and OOB class.
    """
    def __init__(self, num_weak, pred_model, value_estimator, parameters) -> None:
        """
        Args:
            num_weak: B, number of weak learner models
        """
        self.is_debug = parameters.is_debug
        self.discover_step = 0
        self.num_weak=num_weak

        self.saving_path = parameters.saving_path  # '../logs/odvrl/selection_network'

        # Basic RL parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration
        self.val_batch_size = parameters.val_batch_size  # Batch size for validation 
        self.outer_iterations = parameters.epochs

        self.device = parameters.device
        self.num_workers = parameters.num_workers
        self.vest_learning_rate = parameters.vest_learning_rate
        self.pred_learning_rate = parameters.pred_learning_rate
        self.vest_lr_scheduler = parameters.vest_lr_scheduler

        # prepare the original model, prediction model and validation model
        self.ori_model = tf.keras.models.clone_model(pred_model)
        self.pred_model = tf.keras.models.clone_model(pred_model)
        # self.val_model = val_model
        # self.val_model.to(self.device)

        # selection network
        self.value_estimator = value_estimator

        # exploration strategy
        if parameters.explore_strategy == 'constant':
            self.explorer = utils.ConstantExplorer(parameters.epsilon0)
        elif parameters.explore_strategy == 'linear':
            self.explorer = utils.LinearExplorer(parameters.epsilon0)
        elif parameters.explore_strategy == 'exponential':
            self.explorer = utils.ExponentialExplorer(parameters.epsilon0)

    # def _test_acc(self, model, X_val, y_val):
    #     pred_list = []
    #     label_list = []
    #     model.eval()
    #     model.to(self.device)
    #     val_loader = utils.CustomDataloader(X_val, y_val, self.val_batch_size)
    #     for batch_data in val_loader:
    #         feature, label = batch_data

    #         feature = feature.to(self.device)
    #         label = label.to(self.device)

    #         output = model(feature)
    #         _, pred = torch.max(output, 1)

    #         pred = pred.cpu().detach().numpy()
    #         label = label.cpu().detach().numpy()

    #         pred_list += pred.tolist()
    #         label_list += label.tolist()

    #     valid_perf = sum(1 for x, y in zip(pred_list, label_list) if x == y) / len(pred_list)  # accuracy
    #     return valid_perf

    def _calc_oob(self, X, y, idxs):
        rf_model=RandomForestClassifierDV(n_estimators=self.num_weak, n_jobs=-1)
        rf_model.fit(X, y)
        rf_model.evaluate_oob_accuracy(X, y)
        self.oob_cnt[idxs] += rf_model.oob_cnt_for_rl
        self.oob_raw[idxs] += rf_model.oob_raw_for_rl
        return np.divide(self.oob_raw[idxs], self.oob_cnt[idxs])

    def _calc_discover_rate(self, method='proposed'):
        if self.corrupted_num == 0:
            return
        if self.is_debug:
            guess_idxs = np.argsort(np.divide(self.oob_raw, self.oob_cnt))
        else:
            self.discover_step += 1
            values = []
            for i in range(self.num_data // self.val_batch_size):
                part_values = self.evaluate(
                    self.X[i*self.val_batch_size: min((i+1)*self.val_batch_size, self.num_data)], 
                    self.y[i*self.val_batch_size: min((i+1)*self.val_batch_size, self.num_data)]
                )
                values = np.concatenate((values, part_values))
            guess_idxs = np.argsort(values)
        print(guess_idxs[:10])
        print(self.noisy_idxs[:10])
        discover_rate = len(np.intersect1d(guess_idxs[:self.corrupted_num], self.noisy_idxs)) / self.corrupted_num
        wandb.log({'step': self.discover_step, 'discover rate': discover_rate})

    def evaluate(self, X, y):
        # self.val_model.eval()
        label_val_pred = self.val_model.predict(X.numpy())
        X = torch.unsqueeze(X, 1)
        X = X.to(self.device)
        y = y.to(self.device)
        label_one_hot = torch.nn.functional.one_hot(y).to(self.device)
        y_train_hat = torch.abs(label_one_hot - label_val_pred.numpy())
        values =  self.value_estimator(X, label_one_hot, y_train_hat.float())
        values = torch.squeeze(values)
        return values.cpu().detach().numpy()

    def one_step(self, step_id, X, y, X_val, y_val, subset_len, corrupted_num, noisy_idxs, discover_record_interval):
        """Train value estimator, estimate OOB values
        """
        # initialize OOB memory
        self.X = X
        self.y = y
        self.subset_len = subset_len
        self.num_data = len(y)
        self.corrupted_num = corrupted_num
        self.noisy_idxs = noisy_idxs
        self.step_id = step_id
        self.discover_record_interval = discover_record_interval
        self.oob_raw = np.ones(len(y))
        self.oob_cnt = np.ones(len(X))

        self.val_model = tf.keras.models.clone_model(self.pred_model)
        self.val_model.fit(X_val.numpy(), 
                           torch.nn.functional.one_hot(y_val, num_classes=10).numpy())
        self.val_model.verbose = False

        # baseline performance
        # valid_perf = self._test_acc(self.ori_model, X_val, y_val)
        y_valid_hat = self.ori_model.predict(X_val.numpy())
        valid_perf = sklearn.metrics.accuracy_score(y_val, np.argmax(y_valid_hat, axis=1))

        # load evaluation network to device
        self.value_estimator = self.value_estimator.to(self.device)
        # loss function
        dvrl_criterion = RlLoss(self.epsilon).to(self.device)
        freq_criterion = torch.nn.MSELoss()
        # optimizer
        dvrl_optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.vest_learning_rate)
        # learning rate scheduler
        if self.vest_lr_scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(dvrl_optimizer, gamma=0.999)
        elif self.vest_lr_scheduler == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(dvrl_optimizer, factor=1)
        else:
            raise NotImplementedError('this learning rate scheduler is not implemented!')

        best_reward = 0
        for epoch in range(self.outer_iterations):

            # new_model.to(self.device)
            # predictor optimizer
            # pre_optimizer = torch.optim.Adam(new_model.parameters(), lr=self.pred_learning_rate)

            # use a list save all s_input and data values

            for batch_si in (64, 128, 256, 512):
                train_loader = utils.CustomDataloader(X=X, y=y, batch_size=int(batch_si))
                for batch_data in train_loader:
                    # change learning rate
                    scheduler.step()
                    # clean up grads
                    self.value_estimator.train()
                    self.value_estimator.zero_grad()
                    # self.value_estimator.freeze_encoder()
                    dvrl_optimizer.zero_grad()
                    freqsum = 0
                    # train predictor from scratch everytime
                    new_model = tf.keras.models.clone_model(self.pred_model)
                    data_value_list = []
                    s_input = []
                    # self.val_model.eval()
                    # new_model.train()
                    # new_model.zero_grad()
                    # pre_optimizer.zero_grad()

                    feature, label = batch_data
                    # feature = feature.to(self.device)
                    # label = label.to(self.device)
                    label_one_hot = torch.nn.functional.one_hot(label, num_classes=10)
                    label_val_pred = self.val_model(feature.numpy()).numpy()
                    # label_val_pred = label_val_pred.view(label_val_pred.shape[0], -1)

                    y_pred_diff = torch.abs(label_one_hot - label_val_pred)

                    # selection estimation
                    if self.explorer(self.outer_iterations, epoch):
                        est_dv_curr = self._calc_oob(
                            feature.view(feature.shape[0], -1).cpu().detach().numpy(), 
                            label.cpu().detach().numpy(), 
                            train_loader.idxs
                        )
                        est_dv_curr = torch.tensor(est_dv_curr).view(*est_dv_curr.shape, 1)
                        # est_dv_curr = torch.unsqueeze(torch.tensor(est_dv_curr), dim=1)
                        est_dv_curr_hat = self.value_estimator(feature, label_one_hot, y_pred_diff.float())
                        freqsum = freq_criterion(est_dv_curr.float(), est_dv_curr_hat)
                    else:
                        est_dv_curr = self.value_estimator(feature, label_one_hot, y_pred_diff.float())
                        est_dv_curr = est_dv_curr.to('cpu')

                    data_value_list.append(est_dv_curr)

                    # 'sel_prob_curr' is a 01 vector generated by binomial distributions
                    sel_prob_curr = np.random.binomial(1, est_dv_curr.cpu().detach().numpy(), est_dv_curr.shape)
                    # Exception (When selection probability is 0)
                    if np.sum(sel_prob_curr) == 0:
                        est_dv_curr = 0.5 * torch.ones(np.shape(est_dv_curr))
                        sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
                    selected_idxs = np.where(sel_prob_curr)[0]
                    sel_prob_curr = torch.Tensor(sel_prob_curr).to(self.device)
                    s_input.append(sel_prob_curr)

                    # train new prediction model
                    new_model.fit(feature[selected_idxs].numpy(), label_one_hot[selected_idxs].numpy())
                    
                    # output = new_model(feature[selected_idxs].float())
                    # loss function of the prediction model
                    # pre_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    # loss = pre_criterion(output, label[selected_idxs])
                    # back propagation for the prediction model
                    # loss.mean().backward()
                    # pre_optimizer.step()

                    # test the performance of the new model
                    # dvrl_perf = self._test_acc(new_model, X_val, y_val)
                    y_valid_hat = new_model.predict(X_val.numpy())
                    dvrl_perf = sklearn.metrics.accuracy_score(y_val.numpy(), np.argmax(y_valid_hat, axis=1))
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
                    loss = freqsum + dvrl_criterion(data_value_list, s_input, reward)
                    loss.backward()
                    dvrl_optimizer.step()
                    
                    # wandb log
                    wandb.log({'episode': step_id*self.outer_iterations+epoch, 'reward': reward, 'prob': np.max(data_value_list.cpu().detach().numpy())})

                    if epoch % self.discover_record_interval == 0:
                        self._calc_discover_rate()

            if flag_save or epoch % 50 ==0:
                torch.save(
                    self.value_estimator.state_dict(), 
                    os.path.join(
                        self.saving_path, 
                        'net_epoch%d.pth' % (epoch + 1)
                    )
                )
