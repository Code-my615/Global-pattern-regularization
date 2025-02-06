import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, layer_sizes):
        """
        初始化网络
        :param layer_sizes: 每层神经元的数量（列表形式，例如 [2, 100, 100, 1]）
        """
        super(DNN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # 除最后一层外，其他层添加激活函数
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 示例用法
if __name__ == "__main__":

    layer_sizes = [2, 100, 100, 100, 100, 100, 1]
    # 初始化网络
    model = DNN(layer_sizes)

    # 打印网络结构
    print(model)

    # 测试模型
    x = torch.randn(10, layer_sizes[0])
    output = model(x)
    print(output.shape)