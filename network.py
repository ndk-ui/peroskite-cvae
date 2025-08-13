import torch 
from collections import OrderedDict


class LinBlock(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, act, drop_frac):
        super(LinBlock, self).__init__()
        self.layer = torch.nn.Sequential(torch.nn.Linear(num_inputs, num_outputs),
                                         act(),
                                         torch.nn.Dropout(drop_frac))
    def forward(self, x):
        return self.layer(x)

class DNN(torch.nn.Module):
    def __init__(self, sizes, act, drop_frac = 0):
        super(DNN, self).__init__()

        layers = []
        self.depth = len(sizes) - 1 
        for i in range(len(sizes) - 2):
            block = LinBlock(sizes[i], sizes[i+1], drop_frac=drop_frac, act=act)
            layers.append((f'block_{i}', block))
        layers.append(('output', torch.nn.Linear(sizes[-2], sizes[-1])))
        layerDict = OrderedDict(layers) 
        self.layers = torch.nn.Sequential(layerDict) 
    
    def forward(self, x):
        return self.layers(x)