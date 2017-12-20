import os
import torch
import numpy as np
import random

def targetToFloat(target):
    # if target == "positive":
    #     return [0, 0, 1], 2 #[0.0, 0.0, 1.0], 2
    # elif target == "negative":
    #     return [1, 0, 0],  0 #[1.0, 0.0, 0.0],  0
    # else:
    #     return [0, 1, 0], 1 #[0.0, 1.0, 0.0], 1
    if target == "positive":                              #Use Floats for BCELoss, Longs for CrossEntropyLoss
        return [0.0, 0.0, 1.0], 2 
    elif target == "negative":
        return [1.0, 0.0, 0.0],  0 
    else:
        return [0.0, 1.0, 0.0], 1 
    
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word("<pad>")
        self.train, self.train_t, self.train_len, self.tweet_len, self.train_weights = self.tokenize_single(os.path.join(path, 'train.txt'))
        self.valid, self.valid_t, self.valid_len, self.tweet_len, self.valid_weights= self.tokenize_single(os.path.join(path, 'valid.txt'))
        self.test, self.test_t, self.test_len, self.tweet_len, self.test_weights = self.tokenize_single(os.path.join(path, 'test.txt'))

    def tokenize_single(self, path):
        assert os.path.exists(path)
        
        random.seed(1234)

        with open(path, 'r') as f:
            tokens = 0
            tweet_amount = 0
            for line in f:
                target, sentence = line.split(None, 1)
                words = sentence.split()
                tokens += len(words)
                tweet_amount += 1
                for word in words:
                    self.dictionary.add_word(word)

        tweet_len = tokens // tweet_amount

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            targets = torch.FloatTensor(tokens, 3)
            token = 0
            for line in f:
                target, sentence = line.split(None, 1)
                words = sentence.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    this_target, _ = targetToFloat(target)
                    for j in range(3):
                        targets[token][j] = this_target[j]
                    token += 1
                    
        tot = targets.sum(0)
        weights = tot.sum()/tot
        print(path, "tweet_len: ", tweet_len, "tweet_amount ",tweet_amount, "classes", tot) 
    
        return ids, targets, tweet_amount, tweet_len, weights
    
    
    def shuffle_content(self, epoch):
        l = self.tweet_len
        idx = np.arange(self.train_len)
        np.random.seed(epoch)
        np.random.shuffle(idx)
        new_train = torch.LongTensor(self.train_len * self.tweet_len)
        new_train_t = torch.FloatTensor(self.train_len * self.tweet_len, 3)
        for i in range(self.train_len):
            for n in range(l):
                new_train[l * i + n] = self.train[l * idx[i] + n]
                new_train_t[l * i + n] = self.train_t[l * idx[i] + n]
        self.train = new_train
        self.train_t = new_train_t
        