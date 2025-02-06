import numpy as np
import os

def poisson_eq_exact_source(X, coefficient):
    return -(coefficient**2 + coefficient**2)*np.sin(coefficient * X[:, 0]) * np.cos(coefficient * X[:, 1])

def poisson_eq_exact_solution(X, coefficient):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    # 精确解
    return np.sin(coefficient*X[:, 0]) * np.cos(coefficient*X[:, 1])


def gen_poisson_exact_solution(x_num, y_num, x_lower_bound, x_upper_bound, t_lower_bound, t_upper_bound, coefficient):
    """Generates exact solution for the heat equation for the given values of x and t."""
    #这段代码的主要目的是创建一个网格，其中 t 和 x 分别在时间和空间维度上均匀分布。然后，通过创建一个零矩阵 usol，用于存储热方程的精确解。
    #循环遍历所有生成的空间和时间点，并调用 heat_eq_exact_solution 函数获取精确解的值，最后保存为 heat_eq_data.npz.
    # Number of points in each dimension:
    x_dim, y_dim = (x_num, y_num)

    # Create tensors:
    x = np.linspace(x_lower_bound, x_upper_bound, num=x_dim).reshape(x_dim, 1)
    y = np.linspace(t_lower_bound, t_upper_bound, num=y_dim).reshape(y_dim, 1)

    xx, yy = np.meshgrid(x, y)
    x_all, y_all = xx.reshape(-1, 1), yy.reshape(-1, 1)
    X = np.column_stack((x_all, y_all))
    usol = poisson_eq_exact_solution(X, coefficient)
    usol = usol.reshape(y_dim, x_dim)

    return usol