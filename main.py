# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os

import data
import model as base_model
import bidir_lstm as bi_model

parser = argparse.ArgumentParser(description='PyTorch Sentiment Analysis RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/dataset/preprocessed/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, LSTM_BIDIR, LSTM_REV, RAN, RAN_BIDIR)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--nreduced', type=int, default=30,
                    help='number of units in the reduced layer')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--lamb', type=float, default=0.1,
                    help='lambdaL1')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--recallsave', type=str, default='model_recall.pt',
                    help='path to save the final model')
parser.add_argument('--plot', action='store_true',
                    help='plot confusion matrix')
parser.add_argument('--last', action='store_true',
                    help='backpropagate at the end of a tweet')
parser.add_argument('--pre', action='store_true',
                    help='use preprocessed data')
parser.add_argument('--pause', action='store_true',
                    help='not optimise embeddings for the first 5 epochs')
parser.add_argument('--pause_value', type=int, default=0,
                    help='not optimise embeddings for the first 5 epochs')
parser.add_argument('--initial', type=str, default=None,
                    help='path to embedding file. If not set they are initialized randomly')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle train data every epoch')
args = parser.parse_args()

if args.pre:
    args.data = "./data/2017_pre"
    
if args.last:
    dir_name = "LAST_WORD/"
else:
    dir_name = "EVERY_WORD/"
if args.pause_value != 0:
    args.pause = True

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

LEARNING_RATE = args.lr  # 0.005
lambdaL1 = args.lamb

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# print("len of train corpus  ",len(corpus.train))
# print(corpus.train[:20])
# print(corpus.train_t[:20])
# input("Press Enter to continue with batching...")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    # print("batchified dims ",data.size(), " num batch ",nbatch)
    return data

def batchify_target(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, 3).transpose(0, 1).contiguous()
    if args.cuda:
        data = data.cuda()
    # print("batchified dims ",data.size(), " num batch ",nbatch)
    return data

def shuffle_data(epoch):
    corpus.shuffle_content(epoch)
    train_data = batchify(corpus.train, args.batch_size)
    train_data_t = batchify_target(corpus.train_t, args.batch_size)
    return train_data, train_data_t

eval_batch_size = 10
args.bptt = corpus.tweet_len
print("batch size= ",args.batch_size," sequence size= ",args.bptt," tweets number= ",corpus.train.size(0)//corpus.tweet_len,"train len= ",corpus.train.size(0), "train len again= ", corpus.train_len)
# print("corpus ",corpus.train_t)
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
train_data_t = batchify_target(corpus.train_t, args.batch_size)
val_data_t = batchify_target(corpus.valid_t, eval_batch_size)
test_data_t = batchify_target(corpus.test_t, eval_batch_size)
# input("Press Enter to continue with training...")
train_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))
valid_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))
test_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("number of tokens ", ntokens)
if args.model == "LSTM_BIDIR":
    model = bi_model.BI_LSTMModel(args.model, ntokens, args.emsize, args.nhid, args.nreduced, args.nlayers,
                                  args.dropout, args.tied)
elif args.model == "RAN":
    model = base_model.RANModel(args.model, ntokens, args.emsize, args.nhid, 1, args.dropout, args.tied)
elif args.model == "RAN_BIDIR":
    model = bi_model.BI_RANModel(args.model, ntokens, args.emsize, args.nhid, args.nreduced, 1,
                                  args.dropout, args.tied)
else:
    model = base_model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()
    criterionNLLtrain = nn.NLLLoss(weight=corpus.train_weights.cuda())
    criterionNLLvalid = nn.NLLLoss(weight=corpus.valid_weights.cuda())
    criterionNLLtest = nn.NLLLoss(weight=corpus.test_weights.cuda())
else:
    criterionNLLtrain = nn.NLLLoss(weight=corpus.train_weights)
    criterionNLLvalid = nn.NLLLoss(weight=corpus.valid_weights)
    criterionNLLtest = nn.NLLLoss(weight=corpus.test_weights)
criterionBCE = nn.BCELoss()
criterionL1 = nn.L1Loss()

if args.initial is not None:
    model.init_emb_from_file(args.initial)
    

if args.pause:
    model.encoder.weight.requires_grad = False

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def confusion_matrix(output, target, which_matrix):
    _, y = torch.max(output.view(-1, 3), 1)
    _, t = torch.max(target.view(-1, 3), 1)
    t = t.data.cpu().numpy()
    y = y.data.cpu().numpy()
    # print("conf",t,y)
    assert len(t) == len(y), "target and output have different sizes"
    for i in range(len(t)):
        if which_matrix == "training":
            train_confusion[t[i], y[i]] += 1
        elif which_matrix == "validation":
            valid_confusion[t[i], y[i]] += 1
        else:
            test_confusion[t[i], y[i]] += 1
    return


def plotter(which_matrix, epoch=0):
    if which_matrix == "training":
        conf_arr = train_confusion
    elif which_matrix == "validation":
        conf_arr = valid_confusion
    else:
        conf_arr = test_confusion
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = ["negative", "neutral", "positive"]
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + "confusion_matrix_" + which_matrix + "_" + str(epoch) + '.png', format='png')
    plt.close()
    return

def recallFitness(which_matrix):
    if which_matrix == "training":
        conf_arr = train_confusion
    elif which_matrix == "validation":
        conf_arr = valid_confusion
        print(conf_arr)
    else:
        conf_arr = test_confusion
        print(conf_arr)
    recall = np.zeros(3)
    for i in range(len(conf_arr[0])):
        recall[i] = conf_arr[i, i] / (np.sum(conf_arr[i]))
    average_recall = np.sum(recall) / 3
    if which_matrix is not "training":
        print(average_recall)
    return average_recall


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, targets, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(targets[i:i + seq_len, :].view(seq_len, -1, 3))
    return data, target


def evaluate(data_source, targets, test=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    if test:
        criterionNLL = criterionNLLtest
    else:
        criterionNLL = criterionNLLvalid

    if args.model in ["LSTM_BIDIR","RAN_BIDIR"]:
        hidden1, hidden2 = model.init_hidden(eval_batch_size)  # eval_batch_size)
    else:
        hidden = model.init_hidden(eval_batch_size)  # eval_batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        # if len(data_source)-1-i< args.bptt:
        #     continue
        data, targ = get_batch(data_source, targets, i, evaluation=True)

        if args.model in ["LSTM_BIDIR","RAN_BIDIR"]:
            output, hidden1, hidden2 = model(data, hidden1, hidden2)
        else:
            if args.model == "LSTM_REV":
                output, hidden = model(base_model.reverse_input(data, 0), hidden)
            output, hidden = model(data, hidden)

        if args.last:
            last_output = output[-1]
            last_target = targ[-1]
            _, index_target = torch.max(last_target, 1)
            BCE = criterionNLL(last_output, index_target).data
            L1 = criterionL1(last_output, last_target).data
        else:
            _, index_target = torch.max(targ, 2)
            BCE = criterionNLL(output.view(-1, 3), index_target.view(-1)).data
            L1 = criterionL1(output, targ).data

        total_loss += BCE + lambdaL1 * L1

        if args.model in ["LSTM_BIDIR","RAN_BIDIR"]:
            hidden1 = repackage_hidden(hidden1)
            hidden2 = repackage_hidden(hidden2)
        else:
            hidden = repackage_hidden(hidden)

        model.zero_grad()

        if test:
            confusion_matrix(output[-1], targ[-1], "test")
        else:
            confusion_matrix(output[-1], targ[-1], "validation")

    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_BCE = 0
    total_L1 = 0
    start_time = time.time()
    if args.model == "LSTM_BIDIR" or args.model == "RAN_BIDIR":
        hidden1, hidden2 = model.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        # print("training........... ", train_data.size(0)," ", args.bptt)
        optimizer.zero_grad()
        data, targets = get_batch(train_data, train_data_t, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if args.model == "LSTM_BIDIR" or args.model == "RAN_BIDIR":
            hidden1 = repackage_hidden(hidden1)
            hidden2 = repackage_hidden(hidden2)
        else:
            hidden = repackage_hidden(hidden)
        model.zero_grad()
        if args.model == "LSTM_BIDIR"  or args.model == "RAN_BIDIR":
            output, hidden1, hidden2 = model(data, hidden1, hidden2)
        else:
            if args.model == "LSTM_REV":
                output, hidden = model(base_model.reverse_input(data, 0), hidden)
            output, hidden = model(data, hidden)

        last_output = output[-1]
        last_target = targets[-1]

        if args.last:
            _, index_target = torch.max(last_target, 1)
            BCE = criterionNLLtrain(last_output, index_target)
            L1 = criterionL1(last_output, last_target)
        else:
            _, index_target = torch.max(targets, 2)
            BCE = criterionNLLtrain(output.view(-1, 3), index_target.view(-1))
            L1 = criterionL1(output, targets)
        loss = BCE + lambdaL1 * L1
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data
        total_BCE += BCE.data
        total_L1 += L1.data
        

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            cur_BCE = total_BCE[0] / args.log_interval
            cur_L1 = total_L1[0] / args.log_interval
            cur_recall = recallFitness("training") / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:2d}| {:3d}/{:3d}| ms/btc {:4.2f}| '
                  'loss {:5.2f}| BCE {:4.2f}| L1 {:4.2f}| Rec {:3.4f} '.format(
                epoch, batch, len(train_data) // args.bptt,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_BCE, cur_L1, cur_recall))
            total_loss = 0
            total_BCE = 0
            total_L1 = 0
            start_time = time.time()

        confusion_matrix(output[-1], targets[-1], "training")


###############################################################################
# MAIN CODE
###############################################################################

# Loop over epochs.
lr = args.lr
best_val_loss = None
best_epoch = -1
best_recall_epoch = -1
best_fitness = 0
if args.pause:
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
else:
    optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

# At any point you can hit Ctrl + C to break out of training early.
try:
    exec_time = time.time()
    path = "./confusion_matrixes/" + dir_name + args.model + ("_pre" if args.pre else "") + "_lr" + str(
        LEARNING_RATE) + "_lam_" + str(
        lambdaL1) + "_btchsize_" + str(args.batch_size) + "_" + str(exec_time)[-3:] + (
        "_pause" if args.pause else "") + ("_embedd" if args.initial else "") + ("_shuffle/" if args.shuffle else "/")# str(exec_time)

    begin_time = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.pause:
            if epoch > args.pause_value:
                model.encoder.weight.requires_grad=True
                optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

        if args.shuffle:
            # print("...shuffling")
            train_data, train_data_t = shuffle_data(epoch)
            # print("...shuffled!")
            
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, val_data_t)  # evaluate(val_data)
        fitness = recallFitness("validation")
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | loss*100 {:5.2f} | '
              'recall {:3.4f}'.format(epoch, (time.time() - epoch_start_time),
                                      val_loss * 100, fitness))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_epoch = epoch
        if not best_fitness or fitness > best_fitness:
            with open(args.recallsave, 'wb') as f:
                torch.save(model, f)
            best_fitness = fitness
            best_recall_epoch = epoch

        if args.plot:
            plotter("training", epoch)
            plotter("validation", epoch)

        train_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))
        valid_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

end_time = time.time()
print("The best fitness is in Epoch: ", best_epoch)
print("The best recall is in Epoch: ", best_recall_epoch)

###############################################################################
# TEST MODELS
###############################################################################

# Load the best saved model for fitness
with open(args.save, 'rb') as f:
    model = torch.load(f)

test_loss = evaluate(test_data, test_data_t, test=True)
recall_fitness_NLL = recallFitness("test")
print('=' * 89)
print('| Best Loss | Total time {:5.2f}  |  test loss {:5.2f} | test ppl {:8.2f}'.format(
    end_time - begin_time, test_loss, math.exp(test_loss)))
print('=' * 89)
if args.plot:
    plotter("test", epoch=best_epoch)

# Load the best saved model for recall
test_confusion = np.reshape([[0 for i in range(3)] for j in range(3)], (3, 3))
with open(args.recallsave, 'rb') as f:
    model = torch.load(f)

# Run on test data.# Run on test data.
test_loss = evaluate(test_data, test_data_t, test=True)
recall_fitness = recallFitness("test")
print('=' * 89)
print('| Best Recall Average | Total time {:5.2f}  | Recall Fitness {:3.4f}'.format(
    end_time - begin_time, recall_fitness))
print('=' * 89)
if args.plot:
    plotter("test", epoch=best_recall_epoch)

# Print results in txt log file
with open(path + "a_results.txt", 'w') as f:
    f.write(path)
    f.write("\n\nThe best fitness is in Epoch: " + str(best_epoch) + "\nThe best recall is in Epoch: " + str(
        best_recall_epoch))
    f.write(
        '\n\n| Best Recall Average | Total time {:5.2f}  | Recall in Best Fitness {:3.4f} | Recall in Best Recall {:3.4f}'.format(
            end_time - begin_time, recall_fitness_NLL, recall_fitness))
