import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
    
        torch.manual_seed(1234)
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder1 = nn.Embedding(ntoken, ninp)
        self.encoder2 = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM_BIDIR']:
            self.rnn1 = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn2 = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        
        self.decoder = nn.Linear(nhid*2, 3)
        self.softmax = nn.Softmax()
        # self.decoder = nn.Linear(nhid, 1)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        #
        # self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid*2
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden1, hidden2):
        emb1 = self.drop(self.encoder1(input))
        input_reverse = self.reverse_input(input)
        emb2 = self.drop(self.encoder1(input_reverse))
        # print("emb",len(emb))
        # print(emb)
        output1, hidden1 = self.rnn1(emb1, hidden1)
        output2, hidden2 = self.rnn2(emb2, hidden2)
        # print("input ", input)
        # print("output ",output)
        output1 = self.drop(output1)
        output2 = self.drop(output2)
        output = torch.cat((output1, output2), 0)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result_unscaled = self.softmax(decoded)
        result = result_unscaled.view(output.size(0), output.size(1), decoded.size(1))
        # print (result)
        # scaled = self.softmax(result)
        # print("decoded", decoded)
        # print("result ", result)
        return result, hidden1, hidden2

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM_BIDIR':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())),\
                   (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        
    def reverse_input(input):
        idx = [i for i in range(input.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = input.index_select(0, idx)
        return inverted_tensor
