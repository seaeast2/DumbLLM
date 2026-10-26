import torch
import torch.nn as nn


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcurt):
        super().__init__()
        self.use_shortcut = use_shortcurt
        self.layers = nn.ModuleList([ # 5개의 층을 만듭니다.
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x) # 현재 층의 출력을 계산한다.
            if self.use_shortcut and x.shape == layer_output.shape: # 숏컷 연결을 적용할 수 있는지 확인
                x = x + layer_output # 숏컷 연결을 적용
            else:
                x = layer_output # 숏컷 연결을 적용하지 않고 현재 층의 출력을 사용
        return x
