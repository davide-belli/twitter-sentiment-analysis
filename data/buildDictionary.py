# coding: utf-8
import argparse
import numpy as np
import os
import torchwordemb
import gc



parser = argparse.ArgumentParser(description='Builds the word2idx and the filter the embeddings for a dataset')
parser.add_argument('--data', type=str, default='./dataset/preprocessed/',
                    help='location of the data corpus')
parser.add_argument('--embeddings', type=str, default=None,
                    help='choose initialisation(google, path_to_file, None = random)')
parser.add_argument('--save', type=str, default='embeddings.csv',
                    help='path to save the final embeddings')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')

args = parser.parse_args()


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


def fillDictionary(path, dictionary):
    assert os.path.exists(path)
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            target, sentence = line.strip().split(None, 1)
            words = sentence.split()
            
            for word in words:
                dictionary.add_word(word)


def buildEmbMatrixFromFile(fname, vocaboulary, emb_size):
    emb_mat = np.zeros((len(vocaboulary), emb_size))
    
    with open(fname, 'r', encoding='utf-8') as f:
        
        for l in f:
            if l.startswith('#'):
                continue
            word, vec = l.split("\t")
            
            if word in vocaboulary:
                vec = [float(v) for v in vec.split(",")]
                emb_mat[vocaboulary[word]] = np.asarray(vec)
    
    return emb_mat

def buildEmbMatrixFromGoogle(vocaboulary, emb_size):
    print('importing embeddings')
    vocab, vec = torchwordemb.load_word2vec_bin("./GoogleNews-vectors-negative300.bin")
    print('imported embeddings')

    emb_mat = np.zeros((len(vocaboulary), emb_size))

    for i, word in enumerate(vocaboulary.keys()):
        if i % 1000 == 0:
            print("Reading word ", i, "/", len(vocaboulary))
        if word in vocab:
            emb_mat[vocaboulary[word]] = vec[vocab[word]].numpy()
        else:
            emb_mat[vocaboulary[word]] = np.random.normal(0, 1, emb_size)

    print('train matrices built')

    del vec
    del vocab
    gc.collect()

    print('garbage collected')

    return emb_mat

def buildEmbMatrixRandom(vocaboulary, emb_size):
    
    emb_mat = np.zeros((len(vocaboulary), emb_size))

    for i, word in enumerate(vocaboulary.keys()):
        emb_mat[vocaboulary[word]] = np.random.normal(0, 1, emb_size)

    return emb_mat


dictionary = Dictionary()
dictionary.add_word("<pad>")

if not os.path.exists(os.path.join(args.data)):
    os.makedirs(os.path.join(args.data))

for data in ['train', 'test', 'valid']:
    print('Processing', data.upper())
    pre = os.path.join(args.data, data + '.txt')
    fillDictionary(pre, dictionary)



if args.embeddings == "google":
    embeddings = buildEmbMatrixFromGoogle(dictionary.word2idx, args.emsize)
elif args.embeddings is not None:
    if os.path.exists(args.embeddings):
        embeddings = buildEmbMatrixFromFile(args.embeddings, dictionary.word2idx, args.emsize)
    else:
        raise FileNotFoundError("File {} doesn't exist".format(args.embeddings))
else:
    embeddings = buildEmbMatrixRandom(dictionary.word2idx, args.emsize)



save_dir = os.path.dirname(args.save)
if not os.path.exists(save_dir) and save_dir != '':
    os.makedirs(save_dir)

np.savetxt(args.save, embeddings)

