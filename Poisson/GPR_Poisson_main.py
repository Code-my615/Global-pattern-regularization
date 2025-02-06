import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys
from GPR_dataLoader import PoissonDataLoader
from GPR_model import *
from parameters import get_params
from gen_exact_poisson import gen_poisson_exact_solution


parser = argparse.ArgumentParser(description="Train a PINN model by GPR")
parser.add_argument('--experiment', type=str, default='Poisson')
parser.add_argument('--problem', type=str, default='forward')
parser.add_argument("--x_num", type=int, default=30, help="Number of x grid points")
parser.add_argument("--y_num", type=int, default=30, help="Number of y grid points")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--coefficient", type=float, default=1, help="coefficient parameter")
parser.add_argument("--bc_points", type=int, default=400)
parser.add_argument('--lambda_bc', type=int, default=1.0)
parser.add_argument('--lambda_regularization', type=int, default=1.0)
parser.add_argument('--x_truth_grid', type=int, default=256, help='Number of x-coordinates for ground truth grid')  # 测试x轴取的点数
parser.add_argument('--y_truth_grid', type=int, default=100, help='Number of y-coordinates for ground truth grid')  # 测试t轴取的点数
parser.add_argument('--range_x', type=float, nargs=2, default=[-1, 1])
parser.add_argument('--range_y', type=float, nargs=2, default=[-1, 1])
parser.add_argument("--seed", type=int, default=12, help="Random seed")
parser.add_argument('--usol_num', type=int, default=500, help='The number of true solutions used for inverse problems') # 用于逆问题的真实解的个数
parser.add_argument('--layer_sizes', type=int, nargs='+', default=[2, 100, 100, 100, 100, 100, 1])


args = parser.parse_args()
# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 创建目录
save_dir = f"Seed_{args.seed}_{args.experiment}"

# 创建子目录
sub_dir = f"coefficient={args.coefficient}/x_num_{args.x_num},y_num_{args.y_num}/colnum={args.x_num * args.y_num}/lr_{args.lr}/epochs_{args.epochs}"


# sub_sub_dir = sub_dir + '/predict_picture'
base_dir = os.path.join(save_dir, sub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate training dataset
dataset = PoissonDataLoader(args.range_x[0], args.range_y[0], args.range_x[1], args.range_y[1], args.x_num, args.y_num, args.coefficient, args.bc_points, device)

# Processing the test data
x_star = np.linspace(args.range_x[0], args.range_x[1], args.x_truth_grid)
y_star = np.linspace(args.range_y[0], args.range_y[1], args.y_truth_grid)
XX, YY = np.meshgrid(x_star, y_star)
X_test = np.column_stack((XX.reshape(-1, 1), YY.reshape(-1, 1)))
usol = gen_poisson_exact_solution(args.x_truth_grid, args.y_truth_grid, args.range_x[0], args.range_x[1], args.range_y[0], args.range_y[1], args.coefficient)
usol = usol.reshape(-1, 1)
n_x = x_star.shape[0]
n_y = y_star.shape[0]
usol_picture = usol.reshape(n_y, n_x)
# reference solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(YY, XX, usol_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$y$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "reference_solution.png"))
plt.close()


params = get_params(dataset, args, x_star, y_star, usol, device)
if args.problem == 'forward':
    model = GPR_model(params)
else:
    model = Convection_inverse(params)

model.train(args.epochs)
# model.plot_losses(args.epochs, base_dir)
model.text_save(base_dir)

# 预测解
u_pred = model.predict(X_test)
u_pred_picture = u_pred.reshape(n_y, n_x)
l2_error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)
print(l2_error)
# predict solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(YY, XX, u_pred_picture, cmap='jet')
plt.colorbar()
plt.xlabel('$y$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir, "predict_solution.png"))
plt.close()