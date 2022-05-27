import torch
import torch.nn as nn

x = torch.randn(1, 2, 6, 8)  # batch, channel , height , width
print(x)
m = nn.Conv2d(2, 3, (3, 3))  # in_channel, out_channel ,kennel_size,stride
# print(m)
y = m(x)
print(y)
print(y.shape)