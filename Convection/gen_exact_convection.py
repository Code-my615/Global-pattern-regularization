import numpy as np
from ic_function import function
import os

# def function(u0: str):
#     """Initial condition, string --> function."""
#
#     # if u0 == 'sin(x)':
#     #     u0 = lambda x: np.sin(x)
#     # return u0
#     # 将字符串转化为可执行的函数
#     try:
#         return eval(f"lambda x: {str}")
#     except Exception as e:
#         raise ValueError(f"Error parsing function string: {e}")



def convection_diffusion(u0: str, nu, beta, source=0, x_upper_bound = 2 * np.pi, t_upper_bound = 1, xgrid=256, nt=100):
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = xgrid
    h = x_upper_bound / N
    x = np.arange(0, x_upper_bound, h) # not inclusive of the last point
    t = np.linspace(0, t_upper_bound, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals

