import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys
from GPR_dataLoader import ConvectionDataLoader
from GPR_model import *
from parameters import get_params
from gen_exact_convection import convection_diffusion


parser = argparse.ArgumentParser(description="Train a PINN model by GPR")
parser.add_argument('--experiment', type=str, default='convection')
parser.add_argument('--problem', type=str, default='forward')
parser.add_argument("--x_num", type=int, default=84, help="Number of x grid points")
parser.add_argument("--t_num", type=int, default=1200, help="Number of t grid points")
parser.add_argument("--interval", type=int, default=10, help="Interval for constraint layers")
parser.add_argument("--epochs", type=int, default=20000, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--beta", type=float, default=70, help="Beta parameter")
parser.add_argument("--ic_points", type=int, default=256)
parser.add_argument('--lambda_ic', type=int, default=1.0)
parser.add_argument('--lambda_regularization', type=int, default=1.0)
parser.add_argument('--x_truth_grid', type=int, default=256, help='Number of x-coordinates for ground truth grid')  # 测试x轴取的点数
parser.add_argument('--t_truth_grid', type=int, default=100, help='Number of t-coordinates for ground truth grid')  # 测试t轴取的点数
parser.add_argument('--range_x', type=float, nargs=2, default=[0, 2 * np.pi])
parser.add_argument('--range_t', type=float, nargs=2, default=[0, 1])
parser.add_argument("--seed", type=int, default=12, help="Random seed")
parser.add_argument('--usol_num', type=int, default=500, help='The number of true solutions used for inverse problems') # 用于逆问题的真实解的个数
parser.add_argument('--layer_sizes', type=int, nargs='+', default=[2, 100, 100, 100, 100, 100, 1])  # 对流方程中的初始条件
parser.add_argument('--u0_str', type=str, default='sin(x)')  # 对流方程中的初始条件


args = parser.parse_args()
# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 创建目录
save_dir = f"Seed_{args.seed}_{args.experiment}"
# 替换路径中的括号
u0_str = args.u0_str
safe_u0_str = u0_str.replace('(', '_').replace(')', '_').replace('*', '-')
# 创建子目录
sub_dir = f"beta={args.beta}/ic_function={safe_u0_str}/x_num_{args.x_num},t_num_{args.t_num}/colnum={args.x_num * int(args.t_num / args.interval)} {args.x_num} {args.t_num} {args.interval}/lr={args.lr}/epochs_{args.epochs}"


# sub_sub_dir = sub_dir + '/predict_picture'
base_dir = os.path.join(save_dir, sub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate training dataset
dataset = ConvectionDataLoader(args.range_x[0], args.range_t[0], args.range_x[1], args.range_t[1], args.x_num, args.t_num, args.interval, args.beta, args.ic_points, device, args.u0_str)

# Processing the test data
x_star = np.linspace(0, 2 * np.pi, args.x_truth_grid)
t_star = np.linspace(0, 1, args.t_truth_grid)
XX, TT = np.meshgrid(x_star, t_star)  # all the X grid points T times, all the T grid points X times
X_test = np.column_stack((XX.reshape(-1, 1), TT.reshape(-1, 1)))
usol = convection_diffusion(args.u0_str, 0.0, args.beta,
                              x_upper_bound=args.range_x[1],
                              t_upper_bound=args.range_t[1],
                              xgrid=args.x_truth_grid, nt=args.t_truth_grid)
usol = usol.reshape(-1, 1)
n_x = x_star.shape[0]
n_t = t_star.shape[0]
usol_picture = usol.reshape(n_t, n_x)
# reference solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(TT, XX, usol_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "reference_solution.png"))
plt.close()


params = get_params(dataset, args, x_star, t_star, usol, device)
if args.problem == 'forward':
    model = GPR_model(params)
else:
    model = Convection_inverse(params)

model.train(args.epochs)
# model.plot_losses(args.epochs, base_dir)
model.text_save(base_dir)

# 预测解
u_pred = model.predict(X_test)
u_pred_picture = u_pred.reshape(n_t, n_x)
l2_error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)
print(l2_error)
# predict solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(TT, XX, u_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "predict_solution.png"))
plt.close()