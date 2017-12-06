import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from ran import RAN


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nunits, nlayers, dropout=0.5, tie_weights=False):
    
        torch.manual_seed(1234)
        super(RNNModel, self).__init__()
        if rnn_type == 'LSTM_REV':
            rnn_type = 'LSTM'
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emsize, padding_idx=0)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emsize, nunits, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize, nunits, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nunits, 3)
        self.softmax = nn.Softmax()
        # self.decoder = nn.Linear(nunits, 1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nunits != emsize:
                raise ValueError('When using the tied flag, nunits must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nunits = nunits
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        # print("emb",len(emb))
        # print(emb)
        output, hidden = self.rnn(emb, hidden)
        # print("input ", input)
        # print("output ",output)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result_unscaled = self.softmax(decoded)
        result = result_unscaled.view(output.size(0), output.size(1), decoded.size(1))
        # print (result)
        # scaled = self.softmax(result)
        # print("decoded", decoded)
        # print("result ", result)
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nunits).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nunits).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nunits).zero_())

def reverse_input(x, dim):
    # rNpArr = np.flip(x.data.cpu().numpy(), dim).copy()
    # rTensor = torch.from_numpy(rNpArr).cuda()
    # return Variable(rTensor)
    
    input = x
    idx = [i for i in range(input.size(0) - 1, -1, -1)]
    idx = Variable(torch.cuda.LongTensor(idx))
    inverted_tensor = input.index_select(0, idx)
    return inverted_tensor


class RANModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken,  emsize, nunits, nlayers, dropout=0.5, tie_weights=False):
        super(RANModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken,  emsize)
        if rnn_type == "RAN":
            self.rnn = RAN(emsize, nunits, nlayers, dropout=dropout)
        elif rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emsize, nunits, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RAN', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize, nunits, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nunits, 3)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nunits != emsize:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nunits = nunits
        self.nlayers = nlayers

    def init_weights(self):
        init.xavier_uniform(self.encoder.weight)
        self.decoder.bias.data.fill_(0)
        init.xavier_uniform(self.decoder.weight)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nunits).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nunits).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nunits).zero_())
