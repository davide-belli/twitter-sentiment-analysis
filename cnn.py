import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, emb_size, pad_len, vocab_size, classes = 3, channels = 20):
        super(CNN, self).__init__()
        
        self.filters_sizes = [3, 4, 5]
        self.channels = channels
        
        self.emb_size = emb_size
        self.pad_len = pad_len
        self.classes = classes
        self.vocab_size = vocab_size
        
        self.encoder = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        
        conv_layers = []
        for i, filter in enumerate(self.filters_sizes):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(1, self.channels, kernel_size=(self.emb_size, filter), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((1, self.pad_len - filter + 1))
                )
            )
        
        self.conv_layers = nn.ModuleList(conv_layers)

        self.do1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(self.channels * len(self.filters_sizes), 10)

        self.do2 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(10, self.classes)
        
        self.sm = nn.Softmax()
    
    def forward(self, x):
        
        x = self.encoder(x).view((-1, 1, self.emb_size, self.pad_len))
        
        features = []
        
        for conv in self.conv_layers:
            c = conv(x)
            features.append(c.view(c.size(0), -1))
        
        out = torch.cat(features, dim=1)

        # out = out.view(out.size(0), -1)

        out = self.do1(out)

        out = self.fc1(out)

        out = self.do2(out)

        out = self.fc2(out)
        
        out = self.sm(out)
        
        return out.view(1, -1, self.classes)
    
    
    def init_emb_from_file(self, path):
        emb_mat = np.genfromtxt(path)
        self.encoder.weight.data.copy_(torch.from_numpy(emb_mat))
    
    def init_hidden(self, *args):
        return

