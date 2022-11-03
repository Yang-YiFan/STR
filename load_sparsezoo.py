import torch
import torch.nn as nn
from sparseml.pytorch.models.classification.vgg import *
model = vgg16()
#model.load_state_dict(torch.load('vgg16_default.pth')['state_dict'])
model.load_state_dict(torch.load('vgg16_90.pth')['state_dict'])

for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        # sparsity ratio
        print(n, m.weight.size(), 1.0 - 1.0 * torch.count_nonzero(m.weight) / torch.numel(m.weight))
