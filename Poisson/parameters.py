import numpy as np
import os
import sys
import torch


def get_params(dataset, args, x_star, y_star, usol, device):

    params = {
        "problem": args.problem,
        "layer_sizes": args.layer_sizes,
        "train_data":{
            "X_constrain_col": dataset.X_constrain_col,
            "X_constrain_bc": dataset.X_constrain_bc,
            "A_matrix": dataset.A_matrix,
            "B_matrix": dataset.B_matrix,
            "f": dataset.f,
            "tao": dataset.tao,
            "h": dataset.h,
            "X_bc": dataset.X_bc,
            "bc_y_true": dataset.bc_y_true,
            "usol_num": args.usol_num  #用于逆问题的真实解的个数
        },
        "train_params": {
            "optimizer": torch.optim.Adam,
            "lr": args.lr,
            "lambda_bc": args.lambda_bc,
            "lambda_regularization": args.lambda_regularization,
        },
        "viz_params": {
            "x_star": x_star,
            "y_star": y_star,
            "usol": usol,
        },
        "pde_params": {
            "coefficient": args.coefficient,
        },
        "device": device
    }
    return params
