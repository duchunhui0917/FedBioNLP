import torch
import numpy as np

a = torch.LongTensor([[3],
                      [0],
                      [1]])
b = torch.ones_like(a)
print(a | b)
