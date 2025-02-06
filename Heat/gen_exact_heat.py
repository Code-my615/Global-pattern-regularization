import numpy as np
import os

def heat_eq_exact_solution(X, n, L, a):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    # 精确解
    # return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)
    return np.exp(-((n * np.pi / L) ** 2 * a * X[:, 1])) * np.sin(n * np.pi / L * X[:, 0])


def gen_heat_exact_solution(x_num, t_num, n, a, x_upper_bound=2*np.pi, t_upper_bound=1):
    """Generates exact solution for the heat equation for the given values of x and t."""
    #这段代码的主要目的是创建一个网格，其中 t 和 x 分别在时间和空间维度上均匀分布。然后，通过创建一个零矩阵 usol，用于存储热方程的精确解。
    #循环遍历所有生成的空间和时间点，并调用 heat_eq_exact_solution 函数获取精确解的值，最后保存为 heat_eq_data.npz.
    # Number of points in each dimension:
    x_dim, t_dim = (x_num, t_num)

    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.0)
    x_max, t_max = (x_upper_bound, t_upper_bound)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim)
    x = np.linspace(x_min, x_max, num=x_dim)
    xx, tt = np.meshgrid(x, t)
    x_all, t_all = xx.reshape(-1, 1), tt.reshape(-1, 1)
    X = np.column_stack((x_all, t_all))

    usol = heat_eq_exact_solution(X, n, x_upper_bound, a)
    usol = usol.reshape(len(t), len(x))

    return usol