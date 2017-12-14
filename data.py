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
        return [0.0, 0.0, 1.0], 2 #[0.0, 0.0, 1.0], 2
    elif target == "negative":
        return [1.0, 0.0, 0.0],  0 #[1.0, 0.0, 0.0],  0
    else:
        return [0.0, 1.0, 0.0], 1 #[0.0, 1.0, 0.0], 1
    
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
        # self.train, self.train_t, self.valid, self.valid_t, self.test, self.test_t, self.train_len, self.valid_len, self.test_len, self.tweet_len= self.tokenize(os.path.join(path, 'dataset.txt'))

    def tokenize_single(self, path):
        assert os.path.exists(path)
        
        random.seed(1234)

        with open(path, 'r') as f:
            tokens = 0
            tweet_amount = 0
            for line in f:
                target, sentence = line.split("|_|")
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
                target, sentence = line.split("|_|")
                words = sentence.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    this_target, _ = targetToFloat(target)
                    for j in range(3):
                        targets[token][j] = this_target[j]
                    token += 1
                    
        tot = targets.sum(0)
        weights = tot.sum()/tot
        print(path, "tweet_len: ", tweet_len, "tweet_amount ",tweet_amount, "classes", tot) #, "weightsNLL ",weights)
    
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
            # print("original ", self.train[l*idx[i]:l*idx[i]+l])
            # print("shuffled ", new_train[l*i:l*i+l])
            # input("Continue")
        self.train = new_train
        self.train_t = new_train_t
        
        
        
        
        
        
        
        
    ##########  OLD STUFF

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        
        random.seed(1234)
        
        positive=[]
        neutral=[]
        negative=[]
        with open(path, 'r') as f:
            for line in f:
                target, _ = line.split("|_|")
                # targets.append(target)
                if target == "positive":
                    positive.append(line)
                elif target == "negative":
                    negative.append(line)
                else:
                    neutral.append(line)


        tweets_per_cat = min(len(positive), len(neutral), len(negative))
        # print("tweets per cat",tweets_per_cat)
        pos = positive[:tweets_per_cat]
        neu = neutral[:tweets_per_cat]
        neg = negative[:tweets_per_cat]
        random.shuffle(pos)
        random.shuffle(neu)
        random.shuffle(neg)
        
        tweets_count = tweets_per_cat*3
        # dataset=[]
        # for i in range(tweets_per_cat):
        #     dataset.append("positive|_|" + pos[i])
        #     dataset.append("neutral|_|" + neu[i])
        #     dataset.append("negative|_|" + neg[i])
        # print(positive[:tweets_per_cat])

        for line in pos:
            _,sentence=line.split("|_|")
            words = sentence.split()
            for word in words:
                self.dictionary.add_word(word)
        for line in neu:
            _, sentence = line.split("|_|")
            words = sentence.split()
            for word in words:
                self.dictionary.add_word(word)
        for line in neg:
            _, sentence = line.split("|_|")
            words = sentence.split()
            tweet_len = len(words)
            for word in words:
                self.dictionary.add_word(word)
            
        train_amount = tweets_count // 10*8
        valid_amount = tweets_count // 10
        test_amount = tweets_count // 10
        print("Tweets: ",tweets_count, " train ",train_amount," valid ",valid_amount, " test ",test_amount)
        
        x = np.arange(tweets_count//3)
        np.random.shuffle(x)
        train_idx, valid_idx, test_idx = x[:train_amount//3], x[train_amount//3:train_amount//3+valid_amount//3], x[-test_amount//3:]
        # print("idx train",train_idx)
        # print("idx valid", valid_idx)
        # print("idx test", test_idx)
        
        train_dataset= []
        for i in train_idx:
            train_dataset.append(pos[i])
            train_dataset.append(neg[i])
            train_dataset.append(neu[i])
        random.shuffle(train_dataset)
        valid_dataset = []
        for i in valid_idx:
            valid_dataset.append(pos[i])
            valid_dataset.append(neg[i])
            valid_dataset.append(neu[i])
        random.shuffle(valid_dataset)
        test_dataset = []
        for i in test_idx:
            test_dataset.append(pos[i])
            test_dataset.append(neg[i])
            test_dataset.append(neu[i])
        random.shuffle(test_dataset)

        train_ids = torch.LongTensor(train_amount*tweet_len)
        train_targets = torch.FloatTensor(train_amount*tweet_len, 3)  # Float for MSE, Long for CrossEntropy
        token = 0
        for i in range(train_amount):
            # print(i*tweet_len)
            # print(len(words), tweet_len)
            line = train_dataset[i]
            target, sentence = line.split("|_|")
            words = sentence.split()
            # print(len(words), tweet_len)
            for word in words:
                this_target, _ = targetToFloat(target)
                # print(word)
                this_ids = self.dictionary.word2idx[word]
                train_ids[token] = this_ids
                for j in range(len(train_targets[token])):
                    train_targets[token][j] = this_target[j]
                token += 1

        valid_ids = torch.LongTensor(valid_amount*tweet_len)
        valid_targets = torch.FloatTensor(valid_amount*tweet_len, 3)  # Float for MSE, Long for CrossEntropy
        token = 0
        for i in range(valid_amount):
            line = valid_dataset[i]
            target, sentence = line.split("|_|")
            words = sentence.split()
            for word in words:
                this_target, _ = targetToFloat(target)
                valid_ids[token] = self.dictionary.word2idx[word]
                for j in range(len(valid_targets[token])):
                    valid_targets[token][j] = this_target[j]
                token += 1
                
        test_ids = torch.LongTensor(test_amount*tweet_len)
        test_targets = torch.FloatTensor(test_amount*tweet_len, 3)  # Float for MSE, Long for CrossEntropy
        token = 0
        for i in range(test_amount):
            line = test_dataset[i]
            target, sentence = line.split("|_|")
            words = sentence.split()
            for word in words:
                this_target, _ = targetToFloat(target)
                test_ids[token] = self.dictionary.word2idx[word]
                for j in range(len(test_targets[token])):
                    test_targets[token][j] = this_target[j]
                token += 1

        # Tokenize file content
        # with open(path, 'r') as f:
        #     MAX = int(tweets_count * tweet_len // 8.0)
        #     ids = torch.LongTensor(MAX*3)
        #     targets = torch.FloatTensor(MAX*3,3) #Float for MSE, Long for CrossEntropy
        #     # targets = torch.FloatTensor(tweets_count)
        #     token = 0
        #     sentiments = [0,0,0]
        #     # print("MAX ", MAX, tweets_count)
        #     for n,line in enumerate(f):
        #         target, sentence = line.split("|_|")
        #         words = sentence.split()
        #         # targets[n]=targetToFloat(target)
        #         # print(targets[n])
        #         for word in words:
        #             this_target, this_sentiment = targetToFloat(target)
        #             # print(sentiments[this_sentiment],"<",MAX*len(words))
        #             if sentiments[this_sentiment] < MAX:
        #                 ids[token] = self.dictionary.word2idx[word]
        #                 for i in range(len(targets[token])):
        #                     targets[token][i] = this_target[i]
        #                     # print("targets ", targets[token])
        #                 sentiments[this_sentiment] += 1
        #                 token += 1
                    
        #     print("Dataset sentiments: ", sentiments)
        #
        # print("tweet len = ",tweet_len)
        # # for i in ids:
        # #     print(self.dictionary.idx2word[i])
        # #print(ids)

        return train_ids, train_targets, valid_ids, valid_targets, test_ids, test_targets, train_amount, valid_amount, test_amount, tweet_len

    def tokenize_sentence(self, tweet):
        """Tokenizes a string tweet."""
    
        words = tweet.split()
        tweet_len = len(words)
        ids = torch.LongTensor(tweet_len)

        token = 0
        for word in words:
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        print("\ntweet  = ", words)
        print("\ntweet  = ", words)
        # print("w2i = ",ids)
        # for i in ids:
        #     print(self.dictionary.idx2word[i])
        # print(ids)
    
        return ids,  len(ids)
