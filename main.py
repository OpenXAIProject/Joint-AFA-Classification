import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from time import time
import argparse
from model import DFSNet, SetEncoder, MLP, DuelingNet
import csv
import os
import random
import pprint
import copy
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from data_temp import data_load
from environment import Env
import pickle as pk
from environment import multirange
from scipy.special import expit
from time import time

pp = pprint.PrettyPrinter()

def sample(q_val, available, exist, eps=None):
    """ Sample action from q(.|s) specified by q_val.


    Parameters
    ----------
    q_val : 2-D FloatTensor (batch_size x n_actions)
        Q-value
    available : ByteTensor
        Indicator for avaiable action
    exist : ByteTennsor
        Indicator for existing features in the original data.
        To check whether initial state or not
    eps : FloatTensor, optional
        eps for eps-greedy exploration policy


    Returns
    -------
    action : 1-D IntTensor
        chosen action
    max_q_val : 1-D ndarray
        maximum q-value
    """
    assert len(q_val.size()) == 2
    assert available.size() == exist.size()
    N, n_actions = q_val.size()
    assert available.size()[1] + 1 == n_actions
    if eps is not None:
        assert len(eps.size()) == 1 and (eps.size()[0] == 1 or eps.size()[0] == N)
    exploration_prob = torch.ones(N, n_actions, out=q_val.new())

    # At least one feature has to be found
    # In the initial state, stop action is not avaiable
    initial = (1 - torch.eq(available, exist)).long()
    ind = torch.nonzero(initial.sum(dim=1) == 0).squeeze(-1)
    if len(ind) > 0:
        ind = torch.stack([ind, ind.new(len(ind)).fill_(q_val.size()[-1]-1)],
                dim=-1)
        q_val[ind[:, 0], ind[:, 1]] = -np.inf
        exploration_prob[ind[:, 0], ind[:, 1]] = 0

    # Only available features
    if not available.all():
        ind = torch.nonzero(1-available)
        q_val[ind[:, 0], ind[:, 1]] = -np.inf
        exploration_prob[ind[:, 0], ind[:, 1]] = 0

    max_q_val, action = q_val.max(dim=1)
    noise = q_val.new(N).uniform_()
    if eps is not None:
        noise = q_val.new(N).uniform_()
        exploration = noise < eps
        if exploration.any():
            while True:
                random_action = torch.multinomial(
                        exploration_prob[torch.nonzero(exploration)[:, 0]], 1,
                        replacement=True).squeeze()
                action[exploration] = random_action
                nonterminal = action < (n_actions - 1)
                bool_ts = torch.nonzero(exploration.view(-1) & nonterminal.view(-1)).view(-1)
                if len(bool_ts) == 0 or (exploration_prob[bool_ts, action[bool_ts].view(-1)] > 0).all():
                    assert len(bool_ts) == 0 or available[bool_ts, action[bool_ts].view(-1)].all()
                    break
    nonterminal = torch.nonzero(action < (n_actions - 1)).view(-1)
    if len(nonterminal):
        assert available[nonterminal, action[nonterminal].view(-1)].all()

    return action, max_q_val.detach().cpu().numpy()

def binary_cross_entropy_with_logits(input, target, pos_weight=1,
        size_average=True, reduce=True):
    """ calc binary cross entropy with logits


    Parameters
    ----------
    input : 1-D FloatTensor
        logits
    target : 1-D LongTensor
        0 or 1 indicator for binary class
    pos_weight : int, optional
        Unbalanced data handling by using weighted cross entropy loss
    size_average : bool
        If it is false, this func returns 1-D vector


    Returns
    -------
    loss : 0-D or 1-D FloatTensor
        binary cross entropy (averaged if size_average==True
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    l = 1 + (pos_weight - 1) * target
    loss = input - input * target + l * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class StepRunner(object):
    """
    running model for one step,
    geting target q value from target network,
    classification,
    getting final reward from classifier,
    saving and loading model parameter,
    pretraining,
    training with history data
    """
    def __init__(self, model, args):
        self.model = model
        n_features =  model.n_features
        n_actions = model.n_actions
        n_classes = model.n_classes
        old_model = copy.deepcopy(self.model)
        if args.batch_size == 0:
            batch_size = args.nsteps * args.n_envs
        else:
            batch_size = args.batch_size

        def step(inputs, acquired, exist, eps=None, acquired_aux=None):
            length = torch.sum(acquired.long(), 1)
            if n_features + 1 == n_actions:
                available = exist * (1 - acquired) # acquired_aux
            else:
                available = exist * (1 - acquired_aux)
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            available = available.to(args.device)
            exist = exist.to(args.device)
            eps = eps.to(args.device) if eps is not None else None
            p_y_logit, qval, weight = self.model(inputs, length)
            ind, q_a = sample(qval, available, exist, eps=eps)
            return ind, weight, q_a

        def target_q_val(inputs, acquired, exist, actions=None,
                acquired_aux=None):
            '''Target Q-value (bootstrapping)
            for double dqn (choose action with current param and get old val)

            Returns
            -------
            target q-value : FloatTensor
                If actions is not given maximum q-value
            '''
            N = inputs.size()[0]
            length = torch.sum(acquired.long(), 1)
            available = exist * (1 - acquired) if n_features + 1 == n_actions \
                    else exist * (1 - acquired_aux)
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            available = available.to(args.device)
            exist = exist.to(args.device)
            p_y_logit, qval, weight = old_model(inputs, length)

            if actions is not None:
                # double DQN
                return qval[torch.arange(N, out=actions.new()), actions]
            else:
                # vanila Q-learning (maximum Q-value)
                ind, q_a = sample(qval, available, exist)
                return q_a # return Tensor

        def update_target():
            old_model.load_state_dict(self.model.state_dict())

        def classify(inputs, acquired):
            self.model.eval()
            '''Get classifier output (logits) '''
            length = torch.sum(acquired.long(), 1)
            inputs = inputs.to(args.device)
            length = length.to(args.device)
            p_y_logit = self.model(inputs, length)[0]
            self.model.train()
            return p_y_logit

        def calc_final_reward(p_y_logit, labels):
            if n_classes > 2:
                crss_ent = F.cross_entropy(p_y_logit, labels,
                        reduce=False)
            else:
                crss_ent = \
                  binary_cross_entropy_with_logits(p_y_logit.contiguous().view(-1),
                    labels.float(), reduce=False)
            loglikelihood = -crss_ent
            return loglikelihood

        def save(save_path):
            torch.save(model.state_dict(), save_path)

        def load(save_path):
            model.load_state_dict(torch.load(save_path))

        def get_clf_loss(p_y_logit, labels, pos_weight=1):
            if n_classes > 2:
                return F.cross_entropy(p_y_logit, labels.long())
            else:
                return binary_cross_entropy_with_logits(p_y_logit.view(-1),
                    labels.float(), pos_weight=pos_weight)


        def pretrain_step(clf_optimizer, full_obs, full_labels, full_length, pos_weight=1):
            full_obs = full_obs.to(args.device)
            full_length = full_length.to(args.device)
            full_labels = full_labels.to(args.device)
            # clf loss
            p_y_logit, _, weight = self.model(full_obs,full_length)
            clf_loss = get_clf_loss(p_y_logit, full_labels,
                    pos_weight=pos_weight)
            clf_optimizer.zero_grad()
            clf_loss.backward() #retain_graph=True)
            clf_optimizer.step()


        def train_step(obs, acquired, exist, returns, actions, labels, acquired_aux,
                full_obs, full_labels, full_length, iter,
                pos_weight=1,
                policy_optimizer=None, clf_optimizer=None):
            """Get running history and adjust model parameter


            Parameters
            ----------
            policy_optimizer : torch.optim
            clf_optimizer : torch.optim
            obs : FloatTensor
                masked information
            acquired : ByteTensor
                indicator whether acquired or not
            exist : ByteTensor
                indicator whether unmissing feature or not
            returns : FloatTensor
                estimated returns (cumulative reward)
            actions : LongTensor

            labels: LongTensor

            acquired_aux : ByteTensor
                On some dataset a group of features are acquired at once
                This is to handle when action space is not directly mapped to
                feature indicies
            """
            obs = obs.to(args.device)
            labels = labels.to(args.device)
            actions = actions.to(args.device)
            returns = returns.to(args.device)
            if n_actions != n_features + 1:
                acquired_aux = acquired_aux.to(args.device)
            acquired = acquired.to(args.device)
            exist = exist.to(args.device)
            full_obs = full_obs.to(args.device)
            full_length = full_length.to(args.device)
            full_labels = full_labels.to(args.device)

            # acquired_aux
            if n_actions == n_features + 1:
                available = (1 - acquired) * exist
            else:
                available = (1 - acquired_aux) * exist
            length = torch.sum(acquired.long(), 1)

            sampler = BatchSampler(SubsetRandomSampler(
                range(obs.size()[0])), batch_size, drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                indices = indices.to(args.device)
                obs_ = obs[indices]
                labels_ = labels[indices]
                actions_ = actions[indices]
                returns_ = returns[indices]
                acquired_ = acquired[indices]
                if n_actions != n_features + 1:
                    acquired_aux_ = acquired_aux[indices]
                exist_ = exist[indices]
                if n_actions == n_features + 1:
                    available_ = (1 - acquired_) * exist_ # acquired_aux
                else:
                    available_ = (1 - acquired_aux_) * exist_
                length_ = torch.sum(acquired_.long(), 1)
                # policy update
                p_y_logit, qval, weight = self.model(obs_, length_)
                q_a = inputs = qval[torch.arange(len(indices),
                    out=actions.new()), actions_]
                targets = returns_
                if args.done_action_train:
                    inputs = torch.cat((inputs, qval[:, n_actions -1]), dim=0)
                    final_reward = calc_final_reward(p_y_logit, labels_)
                    targets = torch.cat((targets, final_reward.detach()), dim=0)
                policy_loss = F.smooth_l1_loss(inputs, targets.detach())
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                if clf_optimizer is not None:
                    # clf loss
                    if args.complete:
                        obs_ = torch.cat((obs_, full_obs), dim=0)
                        length_ = torch.cat((length_, full_length), dim=0)
                        labels_ = torch.cat((labels_, full_labels), dim=0)
                    p_y_logit, _, weight = self.model(obs_, length_)
                    clf_loss = get_clf_loss(p_y_logit, labels_, pos_weight=pos_weight)
                    clf_optimizer.zero_grad()
                    clf_loss.backward()
                    clf_optimizer.step()
                else:
                    clf_loss = None
        update_target()

        self.step = step
        self.save = save
        self.load = load
        self.classify = classify
        self.pretrain_step = pretrain_step
        self.target_q_val = target_q_val
        self.update_target = update_target
        self.train_step = train_step


def run_n_steps(step_runner, env, n_steps, eps=None, mode='double'):
    ''' Generate history for training
    '''
    n_features = env.n_features
    n_actions = env.n_actions
    n_envs = env.n_envs
    need_aux = (n_actions != n_features + 1)
    mb_obs, mb_exist, mb_acquired, mb_labels = [], [], [], []
    mb_actions, mb_dones,  mb_rewards = [], [], []
    mb_acquired_aux = [] if need_aux else None

    obs = env.var_obs
    for step in range(n_steps):
        acquired = env.var_acquired
        exist = env.var_exist
        labels = env.var_labels
        acquired_aux = env.var_acquired_aux if need_aux else None

        mb_obs.append(obs.clone())
        mb_acquired.append(acquired.clone())
        if need_aux:
            mb_acquired_aux.append(acquired_aux)
        mb_exist.append(exist.clone())
        mb_labels.append(labels.clone())
        # run model

        actions, _, q_a = step_runner.step(obs, acquired,
                exist, eps, acquired_aux=acquired_aux)
        mb_actions.append(actions.cpu())
        # interact with environment
        obs, rewards, dones = env.step(actions)
        mb_rewards.append(rewards.clone())
        mb_dones.append(dones.clone())
    # s_(t+1)
    actions = step_runner.step(obs, acquired, exist,
            acquired_aux=acquired_aux)[0] if mode == 'double' else None
    q_val = step_runner.target_q_val(obs, acquired, exist, actions,
            acquired_aux=acquired_aux)
    # n_steps x batch_size

    # FIXME oh.............
    mb_obs = torch.stack(mb_obs).view(-1, *env.input_size)
    mb_exist = torch.stack(mb_exist).view(-1, n_actions - 1)
    mb_acquired = torch.stack(mb_acquired).view(-1, n_features)
    if need_aux:
        mb_acquired_aux = torch.stack(mb_acquired_aux).view(-1, n_actions - 1)
    mb_labels = torch.stack(mb_labels).view(-1)

    mb_actions = torch.stack(mb_actions)
    mb_rewards = torch.stack(mb_rewards)
    mb_dones = torch.stack(mb_dones)

    # n_steps x n_envs
    mb_returns = []
    R = q_val.cpu() # FIXME
    for i in range(n_steps)[::-1]:
        R = R * (1. - mb_dones[i].float())
        R = R + mb_rewards[i]
        mb_returns.append(R.clone())
    mb_returns = torch.stack(mb_returns[::-1]) # step x n_env
    mb_actions = mb_actions.view(-1)
    mb_returns = mb_returns.view(-1)

    return mb_obs, mb_acquired, mb_exist, mb_returns, \
            mb_actions, mb_labels, mb_acquired_aux

def test(step_runner, env, args, iter=0):
    args_str = pprint.pformat(vars(args))
    n_classes = env.n_classes
    n_envs = env.n_envs
    n_features = env.n_features
    n_actions = env.n_actions
    n_data = env.n_data
    need_aux = (n_actions != n_features + 1)

    obs = env.var_obs
    offset = 0
    inputs = np.zeros((n_data, n_features))
    acquired = np.zeros((n_data, n_features))
    exist = np.zeros((n_data, n_features))
    if need_aux:
        acquired_aux = np.zeros((n_data, n_actions - 1))
    labels = np.zeros(n_data)
    correct = np.zeros(n_data)
    probs = np.zeros(n_data)
    returns = np.zeros(n_data)
    weights = np.zeros((n_data, n_features))
    order = np.zeros((n_data, n_features))
    q_a = np.zeros((n_data, n_actions))
    if n_classes == 2:
        sigmoid = np.zeros(n_data)
    offset = 0

    while offset < n_data:
        acquired_ = env.var_acquired
        acquired_aux = env.var_acquired_aux if need_aux else None
        exist_ = env.var_exist
        labels_ = env.var_labels
        # run model
        actions, weights_, q_a_ = step_runner.step(
                obs, acquired_, exist_, acquired_aux=acquired_aux)
        obs, done_records = env.test_step(actions, q_a_)
        # record
        if done_records[0] is not None and len(done_records[0]):
            tmp = [done_records[i].shape[0] for i in range(7)]
            assert tmp[0] == tmp[1] == tmp[2] == tmp[3] == tmp[4] == tmp[5]
            n_terminal = done_records[0].shape[0]
            from_ = offset
            to_ = offset + n_terminal
            inputs[from_:to_] = done_records[0]
            acquired[from_:to_] = done_records[1]
            labels[from_:to_] = done_records[2]
            correct[from_:to_] = done_records[3]
            probs[from_:to_] = done_records[4]
            returns[from_:to_] = done_records[5]
            if weights_ is not None: weights[from_:to_] = weights_[done_records[6]]
            q_a[from_:to_] = env.q_a[done_records[6]]
            try:
                order[from_:to_] = done_records[7]
            except:
                pass
            if n_classes == 2:
                sigmoid[from_:to_] = done_records[8]
            offset = to_
    assert offset == n_data
    print(args_str)
    print('accuracy', np.mean(correct))
    if n_classes == 2:
        auc = roc_auc_score(labels, sigmoid)
        print('auc', auc)
    print('n_acquired(mean)', np.mean(np.sum(acquired, 1)))
    print('n_acquired(min)', np.amin(np.sum(acquired, 1)))
    print('n_acquired(max)', np.amax(np.sum(acquired, 1)))
    print('n_acquired(med)', np.median(np.sum(acquired, 1)))
    print('picked detail', list(enumerate(np.sum(acquired, 0).astype(int))))
    if weights_ is not None: print('weight', weights.sum(axis=0) / acquired.sum(axis=0))
    print('returns(mean)', np.mean(returns))
    print('returns(min)', np.amin(returns))
    print('returns(max)', np.amax(returns))
    print('returns(med)', np.median(returns))

    if n_classes > 2:
        return inputs, correct, acquired, returns, weights, \
                probs, labels, order, q_a, exist
    return inputs, correct, acquired, returns, weights, \
            probs, labels, order, q_a, exist, auc


def learn(step_runner, args, env, valenv=None, nsteps=5,
        total_steps=int(80e6), lr=7e-4, scheduler='linear', optim=Adam):
    # TODO lr_scheduler
    n_features, n_classes = env.n_features, env.n_classes

    if args.batch_size == 0:
        batch_size = args.nsteps * args.n_envs
    else:
        batch_size = args.batch_size
    mult = 1 if args.dropout else 0
    params = [{'params' : step_runner.model.clf_weight_params,
              'weight_decay' : mult * (1 - args.p) / batch_size},
             {'params' : step_runner.model.clf_bias_params,
              'weight_decay' : mult * 1 / batch_size},
             {'params': step_runner.model.encoder.parameters()}]
    clf_optimizer = optim(params, lr=lr)
    #policy_optimizer = optim(step_runner.model.policy.parameters(),
    #        lr=lr, weight_decay=0)
    policy_optimizer = optim(list(step_runner.model.policy.parameters()) + \
            list(step_runner.model.encoder.parameters()),
            lr=lr, weight_decay=0)
    n = total_steps // (nsteps * env.n_envs)
    update = 0
    if 'test' not in args.message:
      if scheduler == 'linear':
          decay = args.decay_rate * (args.eps_start - args.eps_end) / (n)
      else:
          decay = 0.999
      eps = args.eps_start
      max_score = 0
      ###################
      # Pretraining start
      ###################
      valdata = valenv.data.features
      valexist = np.ones_like(valdata) if valenv.data.exist is None \
              else valenv.data.exist
      vallength = torch.from_numpy(np.sum(valexist.astype(int),
          axis=1)).to(args.device)
      valinput = np.zeros((len(valdata), n_features, n_features + 1)).astype(np.float32)
      x, y = np.where(valexist)
      y_ = multirange(vallength.cpu())
      valinput[x, y_, 0] = valdata[x, y]
      valinput[x, y_, y+1] = 1
      valinput = torch.from_numpy(valinput).to(args.device)
      valtarget = valenv.data.labels
      max_val_score = 0
      print('pretrain')
      pretrain_start = time()
      for pre_i in range(args.pretrain):
          env.reset(first=False)
          if args.pretrain_sample == 'full':
              step_runner.pretrain_step(clf_optimizer,
                      *env.get_current_batch_with_all_features(), pos_weight=env.pos_weight)
          elif args.pretrain_sample == 'random':
              full_obs, full_labels, full_length = env.get_current_batch_with_random_features()
              step_runner.pretrain_step(clf_optimizer, full_obs, full_labels, full_length, pos_weight=env.pos_weight)
          else: # 'both'
              full_obs, full_labels, full_length = env.get_current_batch_with_all_features()
              full_obs_, full_labels_, full_length_ = env.get_current_batch_with_random_features()
              full_obs = torch.cat((full_obs, full_obs_), 0)
              full_labels = torch.cat((full_labels, full_labels_), 0)
              full_length = torch.cat((full_length, full_length_), 0)
              step_runner.pretrain_step(clf_optimizer, full_obs, full_labels, full_length, pos_weight=env.pos_weight)

          if (pre_i + 1) % 10 == 0:
              step_runner.model.eval()
              vallogit, _, weight = step_runner.model(valinput, vallength)
              vallogit = vallogit.detach().cpu().numpy()
              val_score = roc_auc_score(valtarget, expit(vallogit)) if env.n_classes == 2 \
                      else np.mean(valtarget == vallogit.argmax(axis=1))
              #print(pre_i+1, val_score)
              if val_score > max_val_score:
                  print(pre_i+1, val_score, "SAVE")
                  step_runner.save(os.path.join(args.save_path,"pretrained_best.model"))
                  max_val_score = val_score
              step_runner.model.train()
      pretrain_time = time() - pretrain_start
      step_runner.load(os.path.join(args.save_path, "pretrained_best.model"))
      env.reset()
      ###################
      # Pretraining end
      ###################
      for update in range(1, n + 1):
        eps = max(args.eps_end, eps - decay) if scheduler == 'linear' \
                else decay * eps
        eps_ = torch.linspace(0.1 * eps, eps, env.n_envs) if env.n_envs > 2 \
                else torch.FloatTensor([eps])
        run_history = run_n_steps(step_runner, env, nsteps, eps=eps_, mode=args.mode)
        step_runner.train_step(*run_history,
                *env.get_current_batch_with_all_features(),
                update, pos_weight=env.pos_weight,
                policy_optimizer=policy_optimizer,
                clf_optimizer=clf_optimizer)
        if update % args.target_update_freq:
            step_runner.update_target()
        if update % 100 == 0 or update == n:
            step_runner.model.eval()
            print()
            print(update, "/", n)
            print('Current eps', eps)
            if valenv is not None:
                valenv.reset()
                if n_classes == 2:
                    score = test(step_runner, valenv, args, iter=update)[-1]
                else:
                    correct = test(step_runner, valenv, args, iter=update)[1]
                    score = np.mean(correct)
                if score >= max_score and update > (n / 3): # premature..
                    max_score = score
                    main_train_opt = time()
                    step_runner.save(os.path.join(args.save_path,"trained_best.model"))
            step_runner.model.train()


def test_and_record(step_runner, args, env, valenv=None, testenv=None):
    print("=================== Test start ==================")
    fieldnames = ['acc', 'n_acquired_mean', 'n_acquired_min',
    'n_acquired_max', 'n_acquired_med',  'return']
    n_features, n_classes = env.n_features, env.n_classes
    if n_classes == 2:
        fieldnames = ['auc'] + fieldnames
    fieldnames += ["picked_{}".format(i) for i in range(n_features)]
    argsdict = dict(vars(args))
    argsdict.pop('message')
    sorted_argskey = sorted(argsdict.keys())
    field = []
    result = {}
    for prefix, environ in [('val', valenv), ('tr', env), ('ts', testenv)]:
        step_runner.load(os.path.join(args.save_path, 'trained_best.model'))
        environ.reset()
        # trainset
        if environ is None:
            break
        environ.reset()
        tmp = time()
        test_result = test(step_runner, environ, args)
        print(time() - tmp)
        correct = test_result[1]
        acquired = test_result[2]
        returns = test_result[3]
        field += [prefix + '_' + field for field in fieldnames]
        trainresult = {}
        result[prefix + '_acc'] = np.mean(correct)
        result[prefix + '_n_acquired_mean'] = np.mean(np.sum(acquired, 1))
        result[prefix + '_n_acquired_min'] = np.amin(np.sum(acquired, 1))
        result[prefix + '_n_acquired_max'] = np.amax(np.sum(acquired, 1))
        result[prefix + '_n_acquired_med'] = np.median(np.sum(acquired, 1))
        if n_classes == 2:
            result[prefix + '_auc'] = test_result[-1]#auc
            test_result = test_result[:-3]
        for i in range(n_features):
            result[prefix + '_picked_{}'.format(i)] = np.sum(acquired[:, i])
        result[prefix + '_return'] = np.mean(returns)
    field = sorted_argskey + field
    result.update(argsdict)
    file_exists = os.path.isfile(args.csv_path +'.csv')
    with open(args.csv_path + '.csv', 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main(args):
    embedder_hidden_sizes = args.embedder_hidden_sizes
    embedded_dim = args.embedded_dim
    lstm_size = args.lstm_size
    n_shuffle = args.n_shuffle
    clf_hidden_sizes = args.clf_hidden_sizes
    policy_hidden_sizes = args.policy_hidden_sizes
    shared_dim = args.shared_dim
    nsteps= args.nsteps
    n_envs = args.n_envs
    data_type = args.data_type
    r_cost = args.r_cost

    # TODO data load first, classifier defining and declare env
    traindata, valdata, testdata, cost = data_load(data_type=args.data_type,
              random_seed=args.random_seed, cost_from_file=args.cost_from_file)
    if cost is not None:
        r_cost = cost
    input_dim = traindata.n_features + 1
    clf_output_size = traindata.n_classes if traindata.n_classes > 2 else 1
    encoder = SetEncoder(
        input_dim, traindata.n_features,
        embedder_hidden_sizes, embedded_dim, lstm_size, n_shuffle,
        normalize=args.normalize,
        dropout=args.dropout, p=args.p)

    dfsnet = DFSNet(
        encoder=encoder,
        classifier=MLP(lstm_size + embedded_dim, clf_hidden_sizes, clf_output_size,
            dropout=args.dropout, p=args.p,
            batch_norm=args.batchnorm),
        policy=DuelingNet(lstm_size + embedded_dim, policy_hidden_sizes, shared_dim,
            traindata.n_actions))
    dfsnet.to(args.device)
    step_runner = StepRunner(dfsnet, args)
    env = Env(args, n_envs, r_cost, traindata, step_runner.classify)
    valenv = Env(args, n_envs, r_cost, valdata, step_runner.classify)
    testenv = Env(args, n_envs, r_cost, testdata, step_runner.classify)

    env.classify = step_runner.classify
    valenv.classify = step_runner.classify
    testenv.classify = step_runner.classify
    learn_start = time()
    learn(step_runner, args, env, valenv, nsteps=nsteps,
            total_steps=int(5e6), scheduler=args.scheduler)
    learn_elapsed = time() - learn_start
    dfsnet.eval()
    test_and_record(step_runner, args, env, valenv, testenv)
    print(learn_elapsed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--complete', action='store_true', help='train \
            classifier with complete data')
    parser.add_argument(
        '--pretrain', help='pre classifier training',
        type=int, default=10000,
    )
    parser.add_argument(
        '--pretrain_sample', help='',
        type=str, default='both'
    )
    parser.add_argument(
        '--mode', help='double dqn?', type=str, default='double')
    parser.add_argument('--scheduler', help='ent_coef', type=str,
        default='linear')
    parser.add_argument('--dropout', action='store_true', help='Dropout classifier')
    parser.add_argument('--batchnorm', action='store_true', help='batch norm')
    parser.add_argument('--done_action_train', action='store_true', help='done action train')
    parser.add_argument(
        '--data_type', help='data', type=str, default='cube_20_0.3'
    )
    parser.add_argument('--p', help='dropout prob', type=float, default=0)
    parser.add_argument('--group_norm', type=float, default=0,
        help='group_norm regularization param')
    parser.add_argument(
        '--save_dir', help='save directory name',
        type=str, default='result'
    )
    parser.add_argument(
        '--embedder_hidden_sizes', help='embedder',
        type=str, default='[32, 32]'
    )
    parser.add_argument(
        '--clf_hidden_sizes', help='clf mlp size',
        type=str, default='[32, 32]'
    )
    parser.add_argument(
        '--policy_hidden_sizes', help='a2c mlp size',
        type=str, default='[32]'
    )
    parser.add_argument(
        '--shared_dim', help='a2c net shared vertor dim for pi and v',
        type=int, default='16'
    )
    parser.add_argument(
        '--target_update_freq', help='.',
        type=int, default=100
    )
    parser.add_argument('--eps_start', help='.', type=float, default=1.)
    parser.add_argument('--eps_end', help='.', type=float, default=0.1)
    parser.add_argument('--decay_rate', help='.', type=float, default=2)
    parser.add_argument(
        '--n_envs', help='how many episodes simultaneouly?',
        type=int, default=128
    )
    parser.add_argument(
        '--nsteps', help='num of steps for calc return',
        type=int, default=4
    )
    parser.add_argument(
        '--normalize', help='make embedded feature l2 norm to 1',
        type=bool, default=True
    )
    parser.add_argument(
        '--embedded_dim', help='embedded vector dimension',
        type=int, default=16
    )
    parser.add_argument(
        '--lstm_size', help='encoder lstm size',
        type=int, default=16
    )
    parser.add_argument(
        '--n_shuffle', help='n shuffle',
        type=int, default=5
    )
    parser.add_argument(
        '--r_cost', help='cost weight(negative value)',
        default=-0.05, type=float
    )
    parser.add_argument(
        '--cost_from_file', help='whether the cost info is in data csv file or not', type=bool, default=False)
    parser.add_argument(
        '--random_seed', help='random seed',
        type=int, default=123
    )
    parser.add_argument(
        '--batch_size', help='batch size', type=int, default=128
    )
    parser.add_argument('--message', help='message', type=str, default='')
    args = parser.parse_args()

    args.clf_hidden_sizes = eval(args.clf_hidden_sizes)
    args.policy_hidden_sizes = eval(args.policy_hidden_sizes)
    args.embedder_hidden_sizes = eval(args.embedder_hidden_sizes)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.random_seed)
    else:
        args.device = torch.device('cpu')
    args.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            args.save_dir)

    args.save_path = args.data_type +  \
        "_nenv{}_nsteps{}_cost{}_norm{}".format(args.n_envs, args.nsteps,
            args.r_cost, args.normalize) + \
        'eps_start{}end{}decay{}_'.format(args.eps_start, args.eps_end,
                args.decay_rate) + \
        'complete{}_doneactiontrain{}'.format(args.complete,
                args.done_action_train) + \
        'emb' + '_'.join('%03d' % num for num in args.embedder_hidden_sizes + \
                [args.embedded_dim]) + \
        'clf' + '_'.join('%03d' % num for num in args.clf_hidden_sizes) + \
        'policy' + '_'.join('%03d' % num for num in args.policy_hidden_sizes + \
                [args.shared_dim])
    args.save_path = args.save_path + '_batch_size{}'.format(args.batch_size)
    if args.dropout:
        args.save_path = args.save_path + \
        '_dropout{}'.format(args.p)
    if len(args.message) > 0:
        args.save_path += args.message
    args.save_path +='lstm{}_'.format(args.lstm_size) + 'shuffle{}_'.format(args.n_shuffle)
    if args.pretrain:
        args.save_path += '_pretrain{}_{}'.format(args.pretrain,
                args.pretrain_sample)
    if args.batchnorm:
        args.save_path += '_batchnorm'
    args.save_path = os.path.join(args.save_dir, args.save_path)
    args.csv_path = args.save_path
    args.save_path = args.save_path + 'seed{}'.format(args.random_seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    main(args)
