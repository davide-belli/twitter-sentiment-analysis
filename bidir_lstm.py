import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class BI_LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nunits, nreduced, nlayers, dropout=0.5, tie_weights=False):
    
        torch.manual_seed(1234)
        super(BI_LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emsize, padding_idx=0)
        if rnn_type in ['LSTM_BIDIR']:
            self.rnn1 = getattr(nn, "LSTM")(emsize, nunits, nlayers, dropout=dropout)
            self.rnn2 = getattr(nn, "LSTM")(emsize, nunits, nlayers, dropout=dropout)
        
        self.reducer1 = nn.Linear(nunits, nreduced)
        self.reducer2 = nn.Linear(nunits, nreduced)
        self.decoder = nn.Linear(nreduced, 3)
        self.softmax = nn.Softmax()
        # self.decoder = nn.Linear(nunits, 1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nunits != emsize:
        #         raise ValueError('When using the tied flag, nunits must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        #
        # self.init_weights()

        self.rnn_type = rnn_type
        self.nunits1 = nunits
        self.nunits2 = nunits
        self.nreduced = nreduced
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden1, hidden2):
        emb1 = self.drop(self.encoder(inp))
        input_reverse = reverse_input(inp, 0)
        emb2 = self.drop(self.encoder(input_reverse))
        # print("emb",len(emb))
        # print(hidden1)
        output1, hidden1 = self.rnn1(emb1, hidden1)
        output2, hidden2 = self.rnn2(emb2, hidden2)
        # print("input ", input)
        # print("output ",output)
        output1 = self.drop(output1)
        output2 = self.drop(output2)
        
        red1 = self.reducer1(output1)
        red2 = self.reducer2(output2)
        red = red1 + reverse_input(red2, 0)
        
        decoded = self.decoder(red.view(red.size(0)*red.size(1), red.size(2)))
        result_unscaled = self.softmax(decoded)
        result = result_unscaled.view(red.size(0), red.size(1), decoded.size(1))
        # print (result)
        # scaled = self.softmax(result)
        # print("decoded", decoded)
        # print("result ", result)
        return result, hidden1, hidden2

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nunits1).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nunits1).zero_())),\
               (Variable(weight.new(self.nlayers, bsz, self.nunits2).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nunits2).zero_()))
        
def reverse_input(x, dim):
    rNpArr = np.flip(x.data.cpu().numpy(), dim).copy()
    rTensor = torch.from_numpy(rNpArr).cuda()
    return Variable(rTensor)
    
    # dim = x.dim() + dim if dim < 0 else dim
    # inds = tuple(slice(None, None) if i != dim
    #              else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
    #              for i in range(x.dim()))
    # return x[inds]
    
    # idx = [i for i in range(input.size(0) - 1, -1, -1)]
    # idx = torch.cuda.LongTensor(idx)
    # inverted_tensor = input.index_select(0, idx)
    # return inverted_tensor
