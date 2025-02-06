import numpy as np
import os
import sys
import torch


def get_params(dataset, args, x_star, t_star, usol, device):

    params = {
        "problem": args.problem,
        "layer_sizes": args.layer_sizes,
        "train_data":{
            "X_constrain": dataset.X_constrain,
            "x_num": dataset.x_num,
            "all_A_matrix": dataset.all_A_matrix,
            "X_ic": dataset.X_ic,
            "ic_y_true":dataset.ic_y_true,
            "X_bc": dataset.X_bc,
            "bc_y_true": dataset.bc_y_true,
            "usol_num": args.usol_num  #用于逆问题的真实解的个数
        },
        "train_params": {
            "optimizer": torch.optim.Adam,
            "lr": args.lr,
            "lambda_ic": args.lambda_ic,
            "lambda_bc": args.lambda_bc,
            "lambda_regularization": args.lambda_regularization,
        },
        "viz_params": {
            "x_star": x_star,
            "t_star": t_star,
            "usol": usol,
        },
        "pde_params": {
            "alpha": args.alpha,
        },
        "device": device
    }
    return params
