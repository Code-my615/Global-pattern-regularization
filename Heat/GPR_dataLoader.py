import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import sys
from ic_function import function


class HeatDataLoader:
    def __init__(self, x_min, t_min, x_max, t_max, x_num, t_num, interval, alpha, ic_points, bc_points, device, u0: str='sin(x)'):
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
        self.alpha = alpha
        self.ic_points = ic_points
        self.bc_points = bc_points
        self.device = device
        self.ic_function = function(u0)

        # 生成网格点
        self.X, self.lbd = self._gen_grid()

        if self.alpha * self.lbd > 1 / 2:
            print("不满足稳定性条件")
            sys.exit()

        self.A_matrix = self.gen_A_matrix()

        # 生成层索引
        self.layer_index, self.all_A_matrix = self._generate_layer_index()

        # 处理约束点
        self.X_constrain = self._process_constraints()

        # 加载初始条件数据
        self.X_ic, self.ic_y_true = self._load_ic_data()
        # 加载边界条件数据
        self.X_bc, self.bc_y_true = self._load_bc_data()

    def _gen_grid(self):
        # endpoint=False的原因是因为想要间隔为1/x_num和1/t_num。画x_tis-1个点，产生x_num个间隔。
        x = np.linspace(self.x_min, self.x_max, self.x_num, endpoint=False)
        t = np.linspace(self.t_min, self.t_max, self.t_num, endpoint=False)

        x_dis = (self.x_max - self.x_min)/(self.x_num - 1 + 1)
        t_dis = (self.t_max - self.t_min)/(self.t_num - 1 + 1)
        lbd = t_dis / (x_dis * x_dis)
        print(lbd)
        print(self.alpha * lbd)

        if self.alpha * lbd > 1/2:
            print("不满足稳定性条件")
            sys.exit()

        # remove boundary at x=0
        x_noboundary = x[1:]

        X_noboundary, T = np.meshgrid(x_noboundary, t)
        X_noboundary = np.hstack((X_noboundary.flatten()[:, None], T.flatten()[:, None]))
        return X_noboundary, lbd


    def gen_A_matrix(self):
        """ 生成 A 矩阵 """
        A_matrix = np.diag(np.ones(self.x_num - 2) * (self.alpha * self.lbd), k=-1) + np.diag(np.ones(self.x_num - 1) * (1 - 2 * self.alpha * self.lbd),
                                                                           k=0) + np.diag(
            np.ones(self.x_num - 2) * (self.alpha * self.lbd),
            k=1)
        return torch.tensor(A_matrix, dtype=torch.float32, device=self.device)

    def _generate_layer_index(self):
        """ 生成选中的层索引以及所有 A 矩阵幂次 """
        columns = [i for i in range(self.t_num) if i % self.interval == 0]
        all_A_matrix = [torch.matrix_power(self.A_matrix, i) for i in columns[1:]]
        return columns, torch.vstack(all_A_matrix).to(self.device)

    def _process_constraints(self):
        """ 提取需要约束的层并转换为 Tensor """
        X_re = self.X.reshape(-1, self.x_num-1, 2)  # 3D数组，每行对应一层采样点
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

    def _load_bc_data(self):
        """ 加载初始条件数据并转换为 Tensor """
        t_values = np.random.uniform(self.t_min, self.t_max, self.bc_points)
        bc_left = np.column_stack((np.full_like(t_values, self.x_min), t_values))
        bc_right = np.column_stack((np.full_like(t_values, self.x_max), t_values))

        self.X_bc = np.vstack((bc_left, bc_right))
        # Generate initial condition true values
        self.bc_y_true = np.full_like(self.X_bc, 0)

        return torch.tensor(self.X_bc, dtype=torch.float32, device=self.device), torch.tensor(self.bc_y_true,dtype=torch.float32, device=self.device)