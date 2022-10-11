import time
import torch
import torch.nn as nn

# C H W
a = torch.randn(2048,7,7)
pool = nn.AdaptiveAvgPool2d(1)

st = time.time()
b = pool(a)
et = time.time()

print("time: ", et - st)
