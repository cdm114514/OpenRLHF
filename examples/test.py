import torch
import torch.nn as nn

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 2048)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = SimpleModel().cuda()
input_tensor = torch.randn((300, 2048)).cuda()  # 假设输入是随机初始化的

with torch.no_grad():
    for _ in range(100000):
        model(input_tensor)


