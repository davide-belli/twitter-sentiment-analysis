# coding: utf-8
import argparse
import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--tweet', type=str, default='I want to go to the beach with you tomorrow! :) <eos>',
                    help='tweet to predict the sentiment of')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

corpus = data.Corpus(args.data)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

pred_batch_size=1
tok, tweet_size = corpus.tokenize_sentence(args.tweet)
tweet = batchify(tok,pred_batch_size)
args.bptt = tweet_size


def get_batch(source, evaluation=False):
    data = Variable(source[0:args.bptt], volatile=evaluation)
    return data

def predict(data_source):
    # Turn on evaluation mode which disables dropout.
    
    with open('./model.pt', 'rb') as f:
        model = torch.load(f)
    model.eval()
    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    
    hidden = model.init_hidden(pred_batch_size)  # eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data = get_batch(data_source, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(pred_batch_size, -1)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
    return output_flat

sentiment = predict(tweet).view(-1,3)
print("sentiment: ",sentiment)