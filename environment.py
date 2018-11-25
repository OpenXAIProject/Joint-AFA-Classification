import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.optim import Adam
import torch.nn.functional as F
import pickle
import os
import sys


def exist_feature(exist, act2ftr, n_features):
    exist_ftr = np.zeros((exist.shape[0], n_features))
    for key, item in act2frt.items():
        exist_ind = np.where(exists[:, key])[0]
        exist_ind = exist_ind.reshape(-1, 1)
        exist_ftr[exist_ind, item] = 1
    return exist_ftr

def multirange(counts):
    counts = np.asarray(counts)
    # Remove the following line if counts is always strictly positive.
    counts = counts[counts != 0]

    counts1 = counts[:-1]
    reset_index = np.cumsum(counts1)

    incr = np.ones(counts.sum(), dtype=int)
    incr[0] = 0
    incr[reset_index] = 1 - counts1

    # Reuse the incr array for the final result.
    incr.cumsum(out=incr)
    return incr

def softmax(logits):
    max_ = np.max(logits, 1, keepdims=True)
    a0 = logits - max_
    e0 = np.exp(a0)
    z0 = np.sum(e0, 1, keepdims=True)
    softmax = e0 / z0
    return softmax

def cross_entropy(logits, labels):
    logits = logits.astype(np.float64)
    prob = softmax(logits)
    return -np.log(prob[np.arange(len(logits)), labels]), prob

def sigmoid(logits):
    logits = logits.astype(np.float64)
    z = np.where(logits >= 0, np.exp(-logits), np.exp(logits))
    return np.where(logits >= 0, 1 /(1 + z) , z / (1 + z))

def binary_cross_entropy(logits, labels):
    logits = logits.astype(np.float64)
    # follow pytorch implement
    if not (labels.shape == logits.shape):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.shape, input.shape))
    max_val = np.maximum((-logits), 0)
    loss = logits - logits * labels + max_val + np.log(np.exp(-max_val) + np.exp(-logits - max_val))
    p = sigmoid(logits)

    if np.any(loss == np.inf):
        loss[loss == np.inf] = np.maximum(logits[loss==np.inf],
                -logitis[loss==np.inf])
    return loss, p

class Env(object):
    def __init__(self, args, n_envs, r_cost, data, classify_fn):
        #############
        # load data
        #############
        self.n_features = data.n_features
        self.n_classes = data.n_classes
        self.n_envs = n_envs
        self.r_cost = np.array(args.r_cost)
        self.classify = classify_fn
        self.input_size = (self.n_features, self.n_features + 1)
        self.data = data

        self.pos_weight = 1
        if self.n_classes == 2:
            n_neg = len(np.where(data_labels == 0)[0])
            n_pos = len(np.where(data.labels == 1))
            self.pos_weight = n_neg *1. / n_pos

        self.action2features = data.action2features

        # asset for test_steps
        self.n_data = data.features.shape[0]
        self.n_epoch = 0
        assert self.n_data >= self.n_envs
        batch = data.next_batch(self.n_envs)
        self.batch_inputs = batch[0]
        self.batch_exist = batch[2] if batch[2] is not None \
                else np.ones_like(batch[0]).astype(np.uint8)
        self.batch_labels = batch[1]
        self.n_actions = self.batch_exist.shape[1] + 1
        self.batch_acquired = np.zeros((self.n_envs,
            self.n_features)).astype(np.uint8)
        self.batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1),
                dtype=np.float32).astype(np.float32)
        self.batch_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.q_a = np.zeros((self.n_envs, self.n_actions),
                dtype=np.float32)

        # TODO : gpu.. .cuda()
        self.var_inputs = torch.from_numpy(self.batch_inputs)
        self.var_labels = torch.from_numpy(self.batch_labels)
        self.var_exist = torch.from_numpy(self.batch_exist)
        self.var_acquired = torch.from_numpy(self.batch_acquired)
        self.var_obs = torch.from_numpy(self.batch_obs)
        self.var_rewards = torch.from_numpy(self.rewards)

        if self.n_actions != self.n_features + 1:
            self.batch_acquired_aux = np.zeros_like(self.batch_exist)
            self.var_acquired_aux = torch.from_numpy(self.batch_acquired_aux)

    def calc_reward(self, p_y_logit, labels):
        if self.n_classes > 2:
            crss_ent, prob = cross_entropy(p_y_logit, labels)
        else:
            crss_ent, prob = binary_cross_entropy(p_y_logit.reshape(-1),
                labels)
        loglikelihood = -crss_ent
        assert loglikelihood.shape[0] == prob.shape[0]
        return loglikelihood, prob

    def reset(self, first=True, reset_data=None):
        if reset_data is not None:
            self.data = reset_data
            self.n_data = self.data.data.shape[0]
        if first:
            self.data.index = 0
            self.n_epoch = 0
        batch = self.data.next_batch(self.n_envs)
        self.batch_inputs = batch[0]
        self.batch_labels = batch[1]
        self.batch_exist = batch[2] if batch[2] is not None \
                else np.ones_like(batch[0]).astype(np.uint8)

        self.batch_acquired = np.zeros((self.n_envs,
            self.n_features)).astype(np.uint8)
        self.batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1),
                dtype=np.float32).astype(np.float32)
        self.rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.batch_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.q_a = np.zeros((self.n_envs, self.n_actions),
                dtype=np.float32)

        # TODO : gpu.. .cuda() <- not here
        self.var_inputs = torch.from_numpy(self.batch_inputs)
        self.var_labels = torch.from_numpy(self.batch_labels)
        self.var_exist = torch.from_numpy(self.batch_exist)
        self.var_acquired = torch.from_numpy(self.batch_acquired)
        self.var_obs = torch.from_numpy(self.batch_obs)
        self.var_rewards = torch.from_numpy(self.rewards)
        if self.n_actions != self.n_features + 1:
            self.batch_acquired_aux = np.zeros_like(self.batch_exist)
            self.var_acquired_aux = torch.from_numpy(self.batch_acquired_aux)


    def get_current_batch_with_random_features(self):
        sampled = np.random.binomial(1, 0.5, self.batch_exist.shape) * self.batch_exist.astype(int)
        batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1),
                dtype=np.float32).astype(np.float32)
        if self.n_actions == self.n_features + 1:
            length = np.sum(sampled, axis=1)
            x, y = np.where(sampled)
        else:
            acquired = np.zeros((self.batch_exist.shape[0], self.n_features))
            for key, item in self.action2features.items():
                exist_ind = np.where(sampled[:, key])[0]
                exist_ind = exist_ind.reshape(-1, 1)
                acquired[exist_ind, item] = 1
            length = np.sum(acquired.astype(int), axis=1)
            x, y = np.where(acquired)
        y_ = multirange(length)
        batch_obs[x, y_, 0] = self.batch_inputs[x, y]
        batch_obs[x, y_, y+1] = 1

        return torch.from_numpy(batch_obs), self.var_labels, \
                torch.from_numpy(length)


    def get_current_batch_with_all_features(self):
        batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1),
                dtype=np.float32).astype(np.float32)
        if self.n_actions == self.n_features + 1:
            length = np.sum(self.batch_exist.astype(int), axis=1)
            x, y = np.where(self.batch_exist)
        else:
            acquired = np.zeros((self.batch_exist.shape[0], self.n_features))
            for key, item in self.action2features.items():
                exist_ind = np.where(self.batch_exist[:, key])[0]
                exist_ind = exist_ind.reshape(-1, 1)
                acquired[exist_ind, item] = 1
            length = np.sum(acquired.astype(int), axis=1)
            x, y = np.where(acquired)
        y_ = multirange(length)
        batch_obs[x, y_, 0] = self.batch_inputs[x, y]
        batch_obs[x, y_, y+1] = 1
        return torch.from_numpy(batch_obs), self.var_labels, \
                torch.from_numpy(length)

    def _nonterminal_step(self, actions, nonterminal, length):
        if self.n_actions == self.n_features + 1:
            assert np.all(self.batch_acquired[nonterminal, actions[nonterminal]] == 0)
            self.batch_acquired[nonterminal, actions[nonterminal]] = 1
            self.batch_obs[nonterminal, length[nonterminal], 0] = \
                self.batch_inputs[nonterminal, actions[nonterminal]]
            self.batch_obs[nonterminal, length[nonterminal], actions[nonterminal] + 1] = 1
        else:
            assert np.all(self.batch_acquired_aux[nonterminal, actions[nonterminal]] == 0)
            self.batch_acquired_aux[nonterminal, actions[nonterminal]] = 1
            x = np.where(nonterminal)[0]
            if nonterminal.any():
                x_ = np.concatenate([[i] * np.array(self.action2features[actions[i]]).size for i in x])
                y_ = np.concatenate([np.array(self.action2features[actions[i]]).reshape(-1) \
                    for i in x])
                self.batch_acquired[x_, y_] = 1
                len_acq = [np.array(self.action2features[actions[i]]).size for i in
                    x]
                tmp1 = np.concatenate([[length[i]] *
                        np.array(self.action2features[actions[i]]).size for i in x])
                y__ = tmp1 + multirange(len_acq)
                self.batch_obs[x_, y__, 0] = self.batch_inputs[x_, y_]
                self.batch_obs[x_, y__, y_ + 1] = 1

        if self.r_cost.shape:# len(self.r_cost) == self.n_actions - 1:
            self.rewards[nonterminal] = self.r_cost[actions[nonterminal]]
        else:
            #length = self.batch_acquired.sum(axis=-1)
            self.rewards[nonterminal] = self.r_cost

    def step(self, actions):
        actions = actions.cpu().numpy()
        actions = actions.reshape(-1)
        length = np.sum(self.batch_acquired, axis=-1).astype(int)
        nonterminal = actions < self.n_actions - 1
        self._nonterminal_step(actions, nonterminal, length)
        dones = (~nonterminal).astype(np.float32)

        if not np.all(nonterminal):
            p_y_logit = self.classify(
                torch.from_numpy(self.batch_obs[~nonterminal]),
                torch.from_numpy(self.batch_acquired[~nonterminal])
            ).detach().cpu().numpy()

            reward, prob = self.calc_reward(p_y_logit,
                    self.batch_labels[~nonterminal])
            self.rewards[~nonterminal] = reward # loglikelihood
            # update terminating episode to new!! (input, exist, label)
            n_terminal = np.sum(~nonterminal)

            if ((self.data.index + n_terminal) % self.data.n_data) < self.data.index:
                self.n_epoch += 1 # TODO check where to use?
            batch = self.data.next_batch(n_terminal)
            self.batch_exist[~nonterminal] = batch[2] if batch[2] is not None \
                    else np.ones_like(batch[0]).astype(np.uint8)
            self.batch_inputs[~nonterminal] = batch[0]
            self.batch_labels[~nonterminal] = batch[1]
            self.batch_acquired[~nonterminal] = 0
            if self.n_actions != self.n_features + 1:
                self.batch_acquired_aux[~nonterminal] = 0
            self.batch_obs[~nonterminal] = 0
        var_dones = torch.from_numpy((~nonterminal).astype(np.uint8))
        return self.var_obs, self.var_rewards, var_dones


    def test_step(self, actions, q_a=None):
        # run only one epoch
        actions = actions.cpu().numpy()
        actions = actions.reshape(-1)
        length = np.sum(self.batch_acquired, axis=1).astype(int)
        if self.n_actions == self.n_features + 1:
            self.q_a[length == 0] = 0 #initialize
            self.q_a[np.arange(self.n_envs), length] = q_a
        else:
            length_ = np.sum(self.batch_acquired_aux, axis=1).astype(int)
            self.q_a[length_ == 0] = 0
            self.q_a[np.arange(self.n_envs), length_] = q_a
        nonterminal = actions < self.n_actions - 1
        #if self.n_epoch < 1:
        #    pass
        self._nonterminal_step(actions, nonterminal, length)

        if self.n_epoch < 1:
            notignore = np.arange(self.n_envs)
        else:
            notignore = np.delete(np.arange(self.n_envs), self.ignore)
        terminal_ = (~nonterminal)[notignore] # bool indexing

        if not np.any(terminal_):
            inputs = acquired = labels = correct = probs = returns = dones = order = None
            if self.n_actions != self.n_features + 1:
                acquired_aux = None
            if self.n_classes == 2:
                sigmoid_= None
        else:
            n_terminal = np.sum(terminal_)
            p_y_logit = self.classify(
                    torch.from_numpy(self.batch_obs[notignore][terminal_]),
                    torch.from_numpy(self.batch_acquired[notignore][terminal_])
            ).detach().cpu().numpy()
            reward, prob = self.calc_reward(p_y_logit,
                    self.batch_labels[notignore][terminal_])
            tmp = self.batch_obs[notignore][terminal_][:, :, 1:]
            order = tmp.argmax(axis=2) * tmp.max(axis=2)
            inputs = self.batch_inputs[notignore][terminal_]
            acquired = self.batch_acquired[notignore][terminal_]
            labels = self.batch_labels[notignore][terminal_].astype(int)
            if self.n_actions != self.n_features + 1:
                acquired_aux = self.batch_acquired_aux[notignore][terminal_]

            if self.n_classes > 2:
                correct = np.equal(np.argmax(prob, 1),
                        self.batch_labels[notignore][terminal_])
                probs = prob[np.arange(len(np.where(terminal_)[0])),labels]
            else:
                sigmoid_ = prob #sigmoid(p_y_logit.reshape(-1))
                correct = np.equal(prob >= 0.5,
                        self.batch_labels[notignore][terminal_])
                probs = np.where(self.batch_labels[notignore][terminal_],
                    prob, 1 - prob)
            returns = self.batch_returns[notignore][terminal_] + reward
            # loglikelihood
            dones = ~nonterminal
            if self.n_epoch >= 1:
                dones[self.ignore] = False
            self.batch_returns[~nonterminal] = 0
            # update terminal episode to new!! (input, exist, label, obs,
            # acquired)
            if self.n_epoch >= 1:
                tmp, = np.where(~nonterminal)
                self.ignore = sorted(list(set(tmp) | set(self.ignore)))
            else:
                remain = self.data.n_data - self.data.index
                self.n_epoch += (1 if remain <= n_terminal else 0)
                n_update = min(remain, n_terminal)
                batch = self.data.next_batch(n_update)
                x, = np.where(~nonterminal)
                where = x[:n_update]
                if self.n_epoch == 1:
                    self.ignore = x[n_update:]
                self.batch_exist[where] = batch[2] if batch[2] is not None\
                        else np.ones_like(batch[0]).astype(np.uint8)
                self.batch_inputs[where] = batch[0]
                self.batch_labels[where] = batch[1]
                self.batch_acquired[where] = 0
                if self.n_actions != self.n_features + 1:
                    self.batch_acquired_aux[where] = 0
                self.batch_obs[where] = 0

        if self.n_classes > 2:
            return self.var_obs,\
                (inputs, acquired, labels, correct, probs, returns, dones, order)
        else:
            return self.var_obs,\
                (inputs, acquired, labels, correct, probs, returns, dones, order, sigmoid_)
