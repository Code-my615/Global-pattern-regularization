import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import sys
from gen_exact_poisson import poisson_eq_exact_solution

class PoissonDataLoader:
    def __init__(self, x_min, y_min, x_max, y_max, x_num, y_num, coefficient, bc_points, device):
        """
        x_num: x 方向采样点数
        y_num: y 方向采样点数
        device: 计算设备 ('cpu' 或 'cuda')
        dataset_dir: 数据集路径
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x_num = x_num
        self.y_num = y_num
        self.coefficient = coefficient
        self.bc_points = bc_points
        self.device = device

        # 生成网格点
        self.X, self.tao, self.h = self._gen_grid()
        self.lbd = (self.tao*self.tao)/(self.h*self.h)

        self.X_constrain_col, self.X_constrain_bc, self.x_col_num, self.y_col_num = self._gen_col_and_bc()
        self.A_matrix, self.f, self.B_matrix = self._gen_matrix()

        # 加载边界条件数据
        self.X_bc, self.bc_y_true = self._load_bc_data()

    def _gen_grid(self):
        x = np.linspace(self.x_min, self.x_max, self.x_num)
        y = np.linspace(self.y_min, self.y_max, self.y_num)

        tao = (self.x_max - self.x_min) / (self.x_num - 2 + 1)
        h = (self.y_max - self.y_min) / (self.y_num - 2 + 1)

        xx, yy = np.meshgrid(x, y)
        X = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
        X = X.reshape(self.y_num, self.x_num, 2)  # 弄成一行一行的
        X = X[::-1]  # 更改顺序,将点按照推导的顺序放

        return X, tao, h

    def function(self, x, y):
        return -(self.coefficient ** 2 + self.coefficient ** 2) * np.sin(self.coefficient * x) * np.cos(self.coefficient * y)

    def _gen_col_and_bc(self):

        # 找到任意一层的边界点，相当于边界
        choose_bc_index = 0
        x_col_num = self.x_num - 2 * (choose_bc_index + 1)
        y_col_num = self.y_num - 2 * (choose_bc_index + 1)

        X_constrain_bc = []  # 边界点 坐标由小到大
        for i in range(choose_bc_index, self.y_num - choose_bc_index):
            if i == choose_bc_index or i == self.y_num - choose_bc_index - 1:  # 控制边界点第一行和倒数第一行
                for j in range(choose_bc_index, self.x_num - choose_bc_index):  # 控制边界点的第一列和最后一列
                    X_constrain_bc.append(self.X[i][j])
            else:
                X_constrain_bc.append(self.X[i][choose_bc_index])
                X_constrain_bc.append(self.X[i][self.x_num - choose_bc_index - 1])
        X_constrain_bc = np.array(X_constrain_bc)

        choose_col_index = choose_bc_index + 1

        X_constrain_col = []  # 内部点
        for i in range(choose_col_index, self.y_num - choose_col_index):  # 控制内部点的第一行和最后一行
            for j in range(choose_col_index, self.x_num - choose_col_index):
                X_constrain_col.append(self.X[i][j])
        X_constrain_col = np.array(X_constrain_col)  # 转换成narray数组
        # print(X_constrain_col[0:10])

        return X_constrain_col, X_constrain_bc, x_col_num, y_col_num

    def _gen_matrix(self):

        col_num = self.x_col_num * self.y_col_num
        bc_max_num = self.y_col_num + 2
        A = np.zeros((col_num, col_num))
        for i in range(col_num):

            if i % self.x_col_num != 0:  # i被列数整除的位置都没有
                A[i][i - 1] = self.lbd

            if i >= self.x_col_num:  # 前X_constrain_col_column行没有此操作
                A[i][i - self.x_col_num] = 1

            A[i][i] = -1 * (2 + 2 * self.lbd)

            if (i + 1) % self.x_col_num != 0:  # (i+1)被列数整除的位置都没有
                A[i][i + 1] = self.lbd

            if i < (self.y_col_num - 1) * self.x_col_num:  # 最后几行后面没有
                A[i][i + self.x_col_num] = 1
        # print(A)

        f = []
        for i in range(col_num):
            f.append(self.function(self.X_constrain_col[i][0], self.X_constrain_col[i][1]))
        # print(f)
        f = np.array(f).reshape(-1, 1)

        B = np.zeros((col_num, self.X_constrain_bc.shape[0]))

        # 结合图像来理解
        j = bc_max_num  # 表示列下标，初始指向边界的第二行第一个位置
        k = bc_max_num + 2 * self.y_col_num + 1  # 指向边界最后一行第二个位置

        for i in range(col_num):
            if i < self.x_col_num:  # 内部点第一行跟边界的关系
                B[i][i + 1] = -1
            if i % self.x_col_num == 0:  # 内部点第一列跟边界的关系
                B[i][j] = -self.lbd
                j += 1
            if (i + 1) % self.x_col_num == 0:  # 内部点最后一列跟边界的关系
                B[i][j] = -self.lbd
                j += 1
            if i >= (self.y_col_num - 1) * self.x_col_num:  # 内部点最后一行跟边界的关系
                B[i][k] = -1
                k += 1

        return torch.tensor(A, dtype=torch.float32, device=self.device), torch.tensor(f, dtype=torch.float32, device=self.device), torch.tensor(B, dtype=torch.float32, device=self.device)

    def _load_bc_data(self):
        """ 加载边界条件数据并转换为 Tensor """
        # 左右边界
        y_values = np.random.uniform(self.y_min, self.y_max, self.bc_points)
        X_left = np.column_stack((np.full_like(y_values, self.x_min), y_values))
        X_right = np.column_stack((np.full_like(y_values, self.x_max), y_values))

        x_values = np.random.uniform(self.x_min, self.x_max, self.bc_points)
        Y_left = np.column_stack((np.full_like(x_values, self.y_min), x_values))
        Y_right = np.column_stack((np.full_like(x_values, self.y_max), x_values))

        self.X_bc = np.vstack((X_left, X_right, Y_left, Y_right))
        self.bc_y_true = poisson_eq_exact_solution(self.X_bc, self.coefficient)

        return torch.tensor(self.X_bc, dtype=torch.float32, device=self.device), torch.tensor(self.bc_y_true,dtype=torch.float32, device=self.device)