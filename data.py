import os
import torch

def targetToFloat(target):
    if target == "positive":
        return 1.0
    elif target == "negative":
        return -1.0
    else:
        return 0.0
    
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
        self.train,self.train_t = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid,self.valid_t = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test,self.test_t = self.tokenize(os.path.join(path, 'test.txt'))
        

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            tweets = 0
            tweets_len = []
            max_len=0
            for line in f:
                target,sentence = line.split("|_|")
                words = sentence.split() + ['<eos>']
                tokens += len(words)
                tweets += 1
                tweets_len.append(len(words))
                if len(words)>max_len:
                    max_len=len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.cuda.LongTensor(tweets,max_len).zero_()-1
            targets = torch.cuda.FloatTensor(tweets,1) #cuda!!!
            tweet_number=0
            for line in f:
                token = 0
                #ids.append(torch.cuda.LongTensor(tweets_len[tweet_number])) #cuda!!
                target, sentence = line.split("|_|")
                words = sentence.split() + ['<eos>']
                for word in words:
                    ids[tweet_number][token] = self.dictionary.word2idx[word]
                    targets[tweet_number][0] = targetToFloat(target)
                    token += 1
                tweet_number += 1

        return ids,targets

corpus = Corpus("./data")
# print("Corpus train ",corpus.train[:5])
# print("Corpus t target ",corpus.train_t[:5])
print(corpus.train_t.shape)
print(corpus.train[0].shape)

