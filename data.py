import os
import torch

def targetToFloat(target):
    if target == "positive":
        return 1.0, 2
    elif target == "negative":
        return -1.0,  0
    else:
        return 0.0, 1
    
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
        self.train, self.train_t, self.train_len = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid, self.valid_t, self.valid_len = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, self.test_t, self.test_len = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            tweet_len = -1
            tweets_count = 0
            for line in f:
                target, sentence = line.split("|_|")
                # print(len(sentence)," ",sentence)
                words = sentence.split()
                tweet_len = len(words)
                tweets_count+=1
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            MAX = int(tweets_count * tweet_len // 8.0)
            ids = torch.LongTensor(MAX*3)
            targets = torch.FloatTensor(MAX*3)
            # targets = torch.FloatTensor(tweets_count)
            token = 0
            sentiments = [0,0,0]
            print("MAX ", MAX, tweets_count)
            for n,line in enumerate(f):
                target, sentence = line.split("|_|")
                words = sentence.split()
                # targets[n]=targetToFloat(target)
                # print(targets[n])
                for word in words:
                    this_target, this_sentiment = targetToFloat(target)
                    # print(sentiments[this_sentiment],"<",MAX*len(words))
                    if sentiments[this_sentiment] < MAX:
                        ids[token] = self.dictionary.word2idx[word]
                        targets[token] = this_target
                        sentiments[this_sentiment] += 1
                        token += 1
                    
            print("Dataset sentiments: ", sentiments)
    
        print("tweet len = ",tweet_len)
        # for i in ids:
        #     print(self.dictionary.idx2word[i])
        #print(ids)

        return ids, targets, tweet_len

    def tokenize_sentence(self, tweet):
        """Tokenizes a string tweet."""
    
        words = tweet.split()
        tweet_len = len(words)
        ids = torch.LongTensor(tweet_len)

        token = 0
        for word in words:
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        print("tweet  = ", words)
        print("w2i = ",ids)
        # for i in ids:
        #     print(self.dictionary.idx2word[i])
        # print(ids)
    
        return ids,  len(ids)
