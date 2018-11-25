import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F

def weight_xavier_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
    elif type(m) == nn.LSTMCell:
        torch.nn.init.xavier_normal(m.weight_ih)
        torch.nn.init.xavier_normal(m.weight_hh)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True,
            dropout=False, p=0, group_norm=0, batch_norm=False):
        super(MLP, self).__init__()
        self.layers = []
        self.n_features = int(input_size / 2)
        in_size = input_size
        cnt = 0
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if group_norm > 0 and cnt == 0:
                cnt += 1
                self.w0 = self.layers[-1].weight
                print(self.w0.size())
                assert self.w0.size()[1] == input_size
            if batch_norm:
                print("Batchnorm")
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            if dropout: # for classifier
                print("Dropout!")
                assert p > 0 and p < 1
                self.layers.append(nn.Dropout(p=p))
            in_size = hidden_size
        self.layers.append(nn.Linear(in_size, output_size, bias=bias))
        if batch_norm: # FIXME is it good?
            print("Batchnorm")
            self.layers.append(nn.BatchNorm1d(output_size))
        self.layers = nn.ModuleList(self.layers)

        self.output_size = output_size


    def forward(self, x, length=None):
        for layer in self.layers:
            x = layer(x)
        return x


class SetEncoder(nn.Module):
    def __init__(self,
                 input_dim, n_features,
                 embedder_hidden_sizes, embedded_dim,
                 lstm_size, n_shuffle,
                 simple=True, proj_dim=None, normalize=False,
                 dropout=False, p=0):
        # embedder + lstm
        super(SetEncoder, self).__init__()

        self.n_shuffle = n_shuffle
        self.embedder = MLP(input_dim, embedder_hidden_sizes, embedded_dim,
                dropout=dropout, p=p)
        self.lstm = nn.LSTMCell(embedded_dim, lstm_size)
        #self.module_list = nn.ModuleList([self.embedder, self.lstm])
        self.n_features = n_features
        self.normalize = normalize

        self.lstm_size = lstm_size
        self.embedded_dim = embedded_dim

        if not simple:
            assert proj_dim is not None
            self.attention = nn.ModuleList(
                [nn.Linear(lstm_size, proj_dim, bias=False),
                 nn.Linear(embedded_dim, proj_dim, bias=True),
                 nn.Linear(proj_dim, 1, bias=True)]
            )
        elif embedded_dim != lstm_size:
            self.attention = torch.nn.Linear(lstm_size, embedded_dim,
                    bias=False)
            # torch.nn.init.xavier_normal(self.attention.weight)
            # module.apply(weight_xavier_init)

        def _compute_attention_sum(q, m, length):
            # q : batch_size x lstm_size
            # m : batch_size x max(length) x embedded_dim
            assert torch.max(length) == m.size()[1]
            max_len = m.size()[1]
            if simple:
                if q.size()[-1] != m.size()[-1]:
                    q = self.attention(q) # batch_size x embedded_dim
                weight_logit = torch.bmm(m, q.unsqueeze(-1)).squeeze(2) # batch_size x n_features
            else:
                linear_m = self.attention[1]
                linear_q = self.attention[0]
                linear_out = self.attention[2]

                packed = pack(m, list(length), batch_first=True)
                proj_m = PackedSequence(linear_m(packed.data), packed.batch_sizes)
                proj_m, _ = pad(proj_m, batch_first=True)  # batch_size x n_features x proj_dim
                proj_q = linear_q(q).unsqueeze(1) # batch_size x 1 x proj_dim
                packed = pack(F.relu(proj_m + proj_q), list(length), batch_first=True)
                weight_logit = PackedSequence(linear_out(packed.data), packed.batch_sizes)
                weight_logit, _ = pad(weight_logit, batch_first=True) # batch_size x n_features x 1
                weight_logit = weight_logit.squeeze(2)

            # max_len = weight_logit.size()[1]
            indices = torch.arange(0, max_len,
                out=torch.LongTensor(max_len).unsqueeze(0)).cuda()
            # TODO here.. cuda..
            mask = indices < length.unsqueeze(1)#.long()
            weight_logit[1-mask] = -np.inf
            weight = F.softmax(weight_logit, dim=1) # nonzero x max_len
            weighted = torch.bmm(weight.unsqueeze(1), m)
            # batch_size x 1 x max_len
            # batch_size x     max_len x embedded_dim
            # = batch_size x 1 x embedded_dim
            return weighted.squeeze(1), weight  #nonzero x embedded_dim

        self.attending = _compute_attention_sum


    def forward(self, state, length):
        # length should be sorted
        assert len(state.size()) == 3 # batch x n_features x input_dim
                                      # input_dim == n_features + 1
        batch_size = state.size()[0]
        self.weight = np.zeros((int(batch_size), self.n_features))#state.data.new(int(batch_size), self.n_features).fill_(0.)
        nonzero = torch.sum(length > 0).cpu().numpy() # encode only nonzero points
        if nonzero == 0:
            return state.new(int(batch_size), self.lstm_size + self.embedded_dim).fill_(0.)

        length_ = list(length[:nonzero].cpu().numpy())
        packed = pack(state[:nonzero], length_, batch_first=True)

        embedded = self.embedder(packed.data)


        if self.normalize:
            embedded = F.normalize(embedded, dim=1)
        embedded = PackedSequence(embedded, packed.batch_sizes)
        embedded, _ = pad(embedded, batch_first=True) # nonzero x max(length) x embedded_dim

        # define initial state
        qt = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)
        ct = embedded.new(embedded.size()[0], self.lstm_size).fill_(0.)

        ###########################
        # shuffling (set encoding)
        ###########################

        for i in range(self.n_shuffle):
            attended, weight = self.attending(qt, embedded, length[:nonzero])
            # attended : nonzero x embedded_dim
            qt, ct = self.lstm(attended, (qt, ct))

        # TODO edit here!
        weight = weight.detach().cpu().numpy()
        tmp = state[:, :, 1:]
        val, acq = torch.max(tmp, 2) # batch x n_features
        tmp = (val.long() * acq).cpu().numpy()
        #tmp = tmp.cpu().numpy()
        tmp = tmp[:weight.shape[0], :weight.shape[1]]
        self.weight[np.arange(nonzero).reshape(-1, 1), tmp] = weight

        encoded = torch.cat((attended, qt), dim=1)
        if batch_size > nonzero:
            encoded = torch.cat(
                (encoded,
                 encoded.new(int(batch_size - nonzero),
                     encoded.size()[1]).fill_(0.)),
                dim=0
            )
        return encoded



class DuelingNet(nn.Module):
    def __init__(self, encoded_dim, hidden_sizes, shared_dim, n_actions,
            group_norm=0, batch_norm=False):
        super(DuelingNet, self).__init__()
        self.shared = MLP(encoded_dim, hidden_sizes, shared_dim,
                group_norm=group_norm, batch_norm=batch_norm)
        self.pi_net = MLP(shared_dim, [shared_dim], n_actions)
        self.v_net = MLP(shared_dim, [shared_dim], 1)
        self.n_actions = n_actions

    def forward(self, encoded):
        tmp = self.shared(encoded)
        tmp = F.relu(tmp)
        self.adv = self.pi_net(tmp) # batch_size x n_actions
        self.v = self.v_net(tmp) # batch_size x 1

        output = self.v + (self.adv - torch.mean(self.adv, dim=1, keepdim=True))

        return output #torch.cat((self.pi, self.v), 1)


class DFSNet(nn.Module):
    def __init__(self, encoder=None, classifier=None, policy=None):
        # TODO data uncertainty handling
        super(DFSNet, self).__init__()
        self.encoder = encoder  # MLP or set encoder
        self.classifier = classifier
        self.policy = policy
        self.n_features = classifier.n_features if encoder is None else encoder.n_features
        self.n_actions = policy.n_actions
        assert classifier.output_size != 2
        self.n_classes = classifier.output_size if classifier.output_size > 1 else 2

        clf_params = dict(classifier.named_parameters())
        self.clf_weight_params = [clf_params[key] for key in clf_params.keys() \
                if 'weight' in key]
        self.clf_bias_params = [clf_params[key] for key in clf_params.keys() \
                if 'weight' not in key]


    def forward(self, inputs, length):
        sorted_, indices = torch.sort(length, -1, descending=True)
        _, invert = torch.sort(indices)
        assert (length==sorted_[invert]).all()
        inputs = inputs[indices]#.long()] # sort
        inputs = self.encoder(inputs, sorted_)

        q_val = self.policy(inputs)
        self.q_val = q_val[invert]  # if setencoding else q_val
        weight = self.encoder.weight[invert.cpu().numpy()]

        self.p_y_logit = self.classifier(inputs)[invert]
        return self.p_y_logit, self.q_val, weight


if __name__ == '__main__':

    input_dim = 21
    n_features = 20
    embedder_hidden_sizes = [32, 32]
    embedded_dim = 16
    lstm_size = 20
    n_shuffle = 5
    clf_hidden_sizes = [32]
    a2c_hidden_sizes = [32]
    shared_dim = 16
    batch_size = 4
    n_classes=8
    simple=False
    proj_dim=16

    dfsnet = DFSNet(
        SetEncoder(
            input_dim, n_features,
            embedder_hidden_sizes, embedded_dim, lstm_size, n_shuffle,
            simple=simple, proj_dim=proj_dim
        ),
        MLP(lstm_size + embedded_dim, clf_hidden_sizes, n_classes),
        A2CNet(lstm_size + embedded_dim, a2c_hidden_sizes, shared_dim, n_features + 1)
    )

    print(dfsnet)
    dfsnet.apply(weight_xavier_init)
    print(sorted(list(dict(dfsnet.named_parameters()).keys())))


    batch_in = torch.zeros((batch_size, n_features, n_features + 1))
    a = np.arange(n_features * (n_features + 1) * 1.).reshape(n_features, n_features
            + 1)
    vec_1 = torch.FloatTensor(a) # 3 x 2
    a[2] = 0
    vec_2 = torch.FloatTensor(a)
    a[1] = 0
    vec_3 = torch.FloatTensor(a)
    a[0] = 0
    vec_4 = torch.FloatTensor(a)

    print(vec_1.size())

    batch_in[2] = vec_1
    batch_in[1] = vec_2
    batch_in[3] = vec_3
    batch_in[0] = vec_4
    seq_length = [0, 2, 3, 1]
    seq_length = torch.LongTensor(seq_length)
    outputs = dfsnet(batch_in, seq_length)
    print(outputs)

'''
def create_mlp(input_size, hidden_sizes, output_size, bias=True):
    # return sequential model
    layers = []
    in_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_size, hidden_size, bias=bias))
        layers.append(nn.ReLU())
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size, bias=bias))
    model = torch.nn.Sequential(*layers)
    return model
'''

