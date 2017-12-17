import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, vocab_size, classes = 3, channels = 200, nhidd = 30):
        super(CNN, self).__init__()

        torch.manual_seed(1234)
        
        self.filters_sizes = [5, 6, 7]
        self.channels = channels
        
        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.vocab_size = vocab_size
        
        self.encoder = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        
        conv_layers = []
        for i, filter in enumerate(self.filters_sizes):
            conv = nn.Conv2d(1, self.channels, kernel_size=(self.emb_size, filter), padding=0)
            init.xavier_normal(conv.weight)
            conv_layers.append(
                nn.Sequential(
                    conv,
                    nn.ReLU(),
                    nn.MaxPool2d((1, self.pad_len - filter + 1))
                )
            )
        
        self.conv_layers = nn.ModuleList(conv_layers)

        self.do1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.channels * len(self.filters_sizes), nhidd)

        self.do2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(nhidd, self.classes)
        
        self.sm = nn.LogSoftmax(dim=1)

        init.xavier_normal(self.fc1.weight)
        init.xavier_normal(self.fc2.weight)
        
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

        
        
    
    def forward(self, x):
        
        x = self.encoder(x).view((-1, 1, self.emb_size, self.pad_len))
        
        features = []
        
        for conv in self.conv_layers:
            c = conv(x)
            features.append(c.view(c.size(0), -1))
        
        out = torch.cat(features, dim=1)

        out = self.do1(out)

        out = self.fc1(out)

        out = self.do2(out)

        out = self.fc2(out)
        
        out = self.sm(out)
        
        return out.view(1, -1, self.classes)
    
    
    def init_emb_from_file(self, path):
        emb_mat = np.genfromtxt(path)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))
    
    
        
