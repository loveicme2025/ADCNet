import torch
from thop import profile
from models.MPDNet import MPDNet

model = MPDNet()  # 定义好的网络模型
input1 = torch.randn(1, 3, 256, 256)
input2 = torch.randn(1, 3, 256, 256)
flops, params = profile(model, (input1,input2))
print('flops: %.4fG' % (flops / 1e9) )
print('Number of params: %.4fM' % (params / 1e6))