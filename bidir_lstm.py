import torch
import torch.nn as nn
from torch.autograd import Variable
from ran import RAN
from torch.nn import init
import numpy as np
import torchwordemb
import gc


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

        decoded = self.decoder(red.view(red.size(0) * red.size(1), red.size(2)))
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
                Variable(weight.new(self.nlayers, bsz, self.nunits1).zero_())), \
               (Variable(weight.new(self.nlayers, bsz, self.nunits2).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nunits2).zero_()))


def reverse_input(x, dim):
    # rNpArr = np.flip(x.data.cpu().numpy(), dim).copy()
    # rTensor = torch.from_numpy(rNpArr).cuda()
    # return Variable(rTensor)

    input = x
    idx = [i for i in range(input.size(0) - 1, -1, -1)]
    idx = Variable(torch.cuda.LongTensor(idx))
    inverted_tensor = input.index_select(0, idx)
    return inverted_tensor

def reverse_input_nocuda(x, dim):
    # rNpArr = np.flip(x.data.cpu().numpy(), dim).copy()
    # rTensor = torch.from_numpy(rNpArr).cuda()
    # return Variable(rTensor)

    input = x
    idx = [i for i in range(input.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx))
    inverted_tensor = input.index_select(0, idx)
    return inverted_tensor


class BI_RANModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nunits, nreduced, nlayers, dropout=0.5, tie_weights=False):
        super(BI_RANModel, self).__init__()
        torch.manual_seed(1234)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emsize)
        if rnn_type in ["RAN_BIDIR"]:
            self.rnn1 = RAN(emsize, nunits, nlayers, dropout=dropout)
            self.rnn2 = RAN(emsize, nunits, nlayers, dropout=dropout)

        self.reducer1 = nn.Linear(nunits, nreduced)
        self.reducer2 = nn.Linear(nunits, nreduced)

        self.decoder = nn.Linear(nreduced, 3)
        self.softmax = nn.Softmax()

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
        self.nlayers = nlayers
        self.nreduced = nreduced
        self.nunits1 = nunits
        self.nunits2 = nunits

        self.ntoken = ntoken
        self.emsize = emsize

    def init_weights(self):
        init.xavier_uniform(self.encoder.weight)
        self.decoder.bias.data.fill_(0)
        init.xavier_uniform(self.decoder.weight)

    def forward(self, inp, hidden1, hidden2):
        emb1 = self.drop(self.encoder(inp))
        input_reverse = reverse_input_nocuda(inp, 0)
        emb2 = self.drop(self.encoder(input_reverse))
        output1, hidden1 = self.rnn1(emb1, hidden1)
        output2, hidden2 = self.rnn2(emb2, hidden2)
        output1 = self.drop(output1)
        output2 = self.drop(output2)
        red1 = self.reducer1(output1)
        red2 = self.reducer2(output2)
        red = red1 + reverse_input_nocuda(red2, 0)
        decoded = self.decoder(red.view(red.size(0) * red.size(1), red.size(2)))


        result_unscaled = self.softmax(decoded)
        result = result_unscaled.view(red.size(0), red.size(1), decoded.size(1))
        return result, hidden1, hidden2

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return Variable(weight.new(self.nlayers, bsz, self.nunits1).zero_()), \
               Variable(weight.new(self.nlayers, bsz, self.nunits2).zero_())

    def create_emb_matrix(self, vocabulary):
        print('importing embeddings')
        vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")
        print('imported embeddings')

        emb_mat = np.zeros((self.ntoken, self.emsize))

        for word in vocabulary.keys():
            if word in vocab:
                emb_mat[vocabulary[word]] = vec[vocab[word]].numpy()
            else:
                emb_mat[vocabulary[word]] = np.random.normal(0, 1, self.emsize)

        # hypotetically, the one for <unk>
        # emb_mat[-1] = np.random.normal(0, 1, self.emb_size)

        print('train matrices built')

        del vec
        del vocab
        gc.collect()

        print('garbage collected')

        return emb_mat

    def init_emb(self, vocabulary):
        emb_mat = self.create_emb_matrix(vocabulary)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))