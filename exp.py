from ultralytics.nn.modules.block import AAM
import torch

model = AAM(c1=3, h=100, w=100)
x = torch.randn(size=(1, 3, 100, 100))
# print(model)
y = model(x)
print(y.shape)


