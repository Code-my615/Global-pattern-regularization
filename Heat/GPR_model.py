import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import os
import time
from network import DNN

class GPR_model:
    def __init__(self, params):
        self.device = params["device"]

        # Initalize the network
        self.dnn = DNN(params['layer_sizes']).to(self.device)

       # Use optimizers to set optimizer initialization and update functions
        self.lr = params["train_params"]["lr"]
        self.optimizer_fn = params["train_params"]["optimizer"]
        self.optimizer = self.optimizer_fn(self.dnn.parameters(), self.lr)

        # 用于正问题的pde参数/用于逆问题计算L2 error的真实pde参数
        self.pde_alpha = params["pde_params"]["alpha"]

        # 初始化训练数据
        self.X_constrain = params["train_data"]["X_constrain"]
        self.x_num = params["train_data"]["x_num"]
        self.all_A_matrix = params["train_data"]["all_A_matrix"]
        self.X_ic = params["train_data"]["X_ic"]
        self.ic_y_true = params["train_data"]["ic_y_true"]
        self.X_bc = params["train_data"]["X_bc"]
        self.bc_y_true = params["train_data"]["bc_y_true"]

        # 各项损失函数前的系数
        self.lambda_ic = params["train_params"]["lambda_ic"]
        self.lambda_bc = params["train_params"]["lambda_bc"]
        self.lambda_regularization = params["train_params"]["lambda_regularization"]

        # Creating logs
        self.loss_log = []
        self.loss_ic_log = []
        self.loss_bc_log = []
        self.loss_regularization_log = []
        self.abs_err_log = []
        self.rel_l2_err_log = []

        # Regular Grid for visualization
        x_star = params['viz_params']['x_star']
        t_star = params['viz_params']['t_star']
        self.XX, self.TT = np.meshgrid(x_star, t_star)  # all the X grid points T times, all the T grid points X times
        self.X_star, self.T_star = self.XX.flatten(), self.TT.flatten()
        self.X_test = np.column_stack((self.X_star, self.T_star))
        self.usol = params['viz_params']['usol']
        self.n_x = x_star.shape[0]
        self.n_t = t_star.shape[0]

        self.training_time_seconds = None

        # 误差
        self.MAE = None
        self.MSE = None
        self.l1_error = None
        self.l2_error = None

    def neural_net(self, X):
        u = self.dnn(X)
        return u

    def train(self, epochs):
        self.dnn.train()
        pbar = trange(epochs)
        start_time = time.time()
        for epoch in pbar:

            self.optimizer.zero_grad()

            u_pre_ic = self.neural_net(self.X_ic)
            loss_ic = torch.mean((u_pre_ic - self.ic_y_true) ** 2)

            u_pre_bc = self.neural_net(self.X_bc)
            loss_bc = torch.mean((u_pre_bc - self.bc_y_true) ** 2)

            U_constrain = self.neural_net(self.X_constrain)
            u1 = U_constrain[0:self.x_num-1]
            u_other = U_constrain[self.x_num-1:]
            loss_regularization = torch.mean((torch.mm(self.all_A_matrix, u1) - u_other) ** 2)

            loss = self.lambda_ic * loss_ic + self.lambda_bc * loss_bc + self.lambda_regularization * loss_regularization

            self.loss_regularization_log.append(loss_regularization.item())
            self.loss_bc_log.append(loss_bc.item())
            self.loss_ic_log.append(loss_ic.item())
            self.loss_log.append(loss.item())

            loss.backward()
            self.optimizer.step()

            # 如果想看每轮在测试集上的表现，取消注释。注意会增加一些时间
            # u_pred = self.predict(self.X_test)
            # self.MAE = np.abs(u_pred - self.usol).mean()
            # self.MSE = ((u_pred - self.usol) ** 2).mean()
            # self.l1_error = self.MAE / np.abs(self.usol).mean()
            # self.l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol)
            #
            # self.abs_err_log.append(self.l1_error)
            # self.rel_l2_err_log.append(self.l2_error)

            if (epoch + 1) % 1000 == 0:
                # print("[alpha: %d], Epoch: %d ,loss: %.16f, l2_error: %.16f" % (self.pde_alpha, (epoch+1), loss, self.l2_error))
                print("[alpha: %d], Epoch: %d ,loss: %.16f" % (self.pde_alpha, (epoch + 1), loss))

        end_time = time.time()
        # 计算训练时间
        self.training_time_seconds = end_time - start_time

        # 为记录结果做准备
        u_pred = self.predict(self.X_test)
        self.MAE = np.abs(u_pred - self.usol).mean()
        self.MSE = ((u_pred - self.usol) ** 2).mean()
        self.l1_error = self.MAE / np.abs(self.usol).mean()
        self.l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol)

    def predict(self, X):
        self.dnn.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        u = self.neural_net(X)
        u = u.detach().cpu().numpy()
        return u

    def plot_losses(self, epochs, target_dir, labels=["total Loss", "BC Loss", "IC Loss", "PDE Loss"]):
        x = np.linspace(0, epochs, epochs, False)
        loss_lists = []
        loss_lists.append(self.loss_log)
        loss_lists.append(self.loss_bc_log)
        loss_lists.append(self.loss_ic_log)
        loss_lists.append(self.loss_pde_log)
        for i, loss_list in enumerate(loss_lists):
            plt.plot(x, loss_list, label=labels[i] if labels else f'Loss {i + 1}')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{labels[i]}_epoch-Loss{i + 1}")
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{target_dir}/plot_{labels[i]}_epoch_{epochs}.png")
            # plt.show()
            # plt.close()

    def text_save(self, target_dir):  # filename为写入CSV文件的路径，data为要写入数据列表.
        # 文件路径（将文件命名为 'res'）
        file_path = os.path.join(target_dir, "res")

        # 构建数据字典
        data_dict = {
            "MAE": self.MAE,
            "MSE": self.MSE,
            "L1RE": self.l1_error,
            "L2RE": self.l2_error,
            "Training Time(s)": self.training_time_seconds,
        }

        # 打开文件并写入数据
        with open(file_path, 'a') as file:
            for key, value in data_dict.items():
                file.write(f"{key}: {value:.10f} ")
                if key == "L2RE":
                    file.write('\n')
        file.close()

        with open(f"{target_dir}/pinn_l2_error.txt", 'w') as file:
            for epoch, error in enumerate(self.rel_l2_err_log, 1):
                file.write(f"{epoch},{error}\n")
        file.close()


