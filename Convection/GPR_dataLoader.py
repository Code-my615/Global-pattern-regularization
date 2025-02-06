import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import sys
from ic_function import function


class ConvectionDataLoader:
    def __init__(self, x_min, t_min, x_max, t_max, x_num, t_num, interval, beta, ic_points, device, u0: str='sin(x)'):
        """
        x_num: x 方向采样点数
        t_num: t 方向采样点数
        interval: 多少层进行约束
        beta: 物理参数
        device: 计算设备 ('cpu' 或 'cuda')
        dataset_dir: 数据集路径
        """
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max

        self.x_num = x_num
        self.t_num = t_num
        self.interval = interval
        self.beta = beta
        self.ic_points = ic_points
        self.device = device
        self.ic_function = function(u0)

        # 生成网格点
        self.X, self.lbd = self._gen_grid()

        # 稳定性检查
        if self.beta * self.lbd > 1:
            print(f"不满足稳定性条件: beta * lbd = {self.beta * self.lbd}")
            sys.exit()
        else:
            print(f"满足稳定性条件: beta * lbd = {self.beta * self.lbd}")

        self.A_matrix = self.gen_A_matrix()

        # 生成层索引
        self.layer_index, self.all_A_matrix = self._generate_layer_index()

        # 处理约束点
        self.X_constrain = self._process_constraints()

        # 加载初始条件数据
        self.X_ic, self.ic_y_true = self._load_ic_data()


    def _gen_grid(self):
        """ 生成网格点 """
        x = np.linspace(self.x_min, self.x_max, self.x_num)
        t = np.linspace(self.t_min, self.t_max, self.t_num)

        x_dis = (self.x_max - self.x_min) / (self.x_num - 1)
        t_dis = (self.t_max - self.t_min) / (self.t_num - 1)
        lbd = t_dis / x_dis

        X, T = np.meshgrid(x, t)
        X = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        return X, lbd


    def gen_A_matrix(self):
        """ 生成 A 矩阵 """
        A = np.zeros((self.x_num, self.x_num))
        last_col = self.x_num - 1

        # 周期性边界条件
        A[0][last_col] = 1 - self.beta * self.lbd
        A[0][last_col - 1] = self.beta * self.lbd
        A[last_col] = A[0]

        # 内部行赋值
        for i in range(1, last_col):
            A[i][i - 1] = self.beta * self.lbd
            A[i][i] = 1 - self.beta * self.lbd

        return torch.tensor(A, dtype=torch.float32).to(self.device)

    def _generate_layer_index(self):
        """ 生成选中的层索引以及所有 A 矩阵幂次 """
        columns = [i for i in range(self.t_num) if i % self.interval == 0]
        all_A_matrix = [torch.matrix_power(self.A_matrix, i) for i in columns[1:]]
        return columns, torch.vstack(all_A_matrix).to(self.device)

    def _process_constraints(self):
        """ 提取需要约束的层并转换为 Tensor """
        X_re = self.X.reshape(-1, self.x_num, 2)  # 3D数组，每行对应一层采样点
        X_constrain_column = X_re[self.layer_index, :, :]  # 选出某层采样点
        X_constrain = X_constrain_column.reshape(-1, 2)  # 变成 2D 结构
        # zero = np.zeros((self.x_num * (len(self.layer_index) - 1), 1))  # 一层x_num-1个点，除了u1有len(layer_index)-1层

        # 转换为 Tensor
        return torch.tensor(X_constrain, dtype=torch.float32).to(self.device)

    def _load_ic_data(self):
        """ 加载初始条件数据并转换为 Tensor """
        x_values = np.random.uniform(self.x_min, self.x_max, self.ic_points)
        self.initial = np.column_stack((x_values, np.zeros_like(x_values)))

        # Generate initial condition true values
        self.ic_y_true = self.ic_function(self.initial[:, 0]).reshape(-1, 1)

        return torch.tensor(self.initial, dtype=torch.float32, device=self.device), torch.tensor(self.ic_y_true, dtype=torch.float32, device=self.device)