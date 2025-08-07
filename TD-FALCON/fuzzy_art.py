import torch
from algorithm.utils import fuzzy_and

class FuzzyART:
    def __init__(self, rhos: float, rhol, alpha: float, beta: float, gama: float, device: str):
        """Initialize the Fuzzy ART model.

                Parameters
                ----------
                rho : float
                    Vigilance parameter.
                alpha : float
                    Choice parameter.
                beta : float
                    Learning rate.

                """
        self.params = {
            "rhos": rhos,
            "rhol": rhol,
            "alpha": alpha,
            "beta": beta,
            "gama": gama
        }

        self.n_cluster = 1
        self.input_dim = None
        self.W = None

        self.device = device

    def init_fuzzy_art_net(self, dim):
        self.input_dim = dim
        self.W = torch.ones(self.n_cluster, self.input_dim, dtype=torch.float32, device=self.device)

    def calculate_choice_function(self, x):
        up = torch.sum(fuzzy_and(x, self.W), dim=1)
        down = self.params["alpha"] + torch.sum(self.W, dim=1)
        return self.params["gama"] * (up / down)

    def judge_match(self, x, rho):
        up = torch.sum(fuzzy_and(x, self.W), dim=1)
        down = torch.sum(x, dim=0)
        mkj = up / down
        mkj = mkj.cpu().numpy()
        match = [i >= self.params[rho] for i in mkj]
        return match

    def updatew(self, x, idx):
        self.W[idx] = (1 - self.params["beta"]) * self.W[idx] + self.params["beta"] * fuzzy_and(x, self.W[idx])

    def add_cluster(self, x, idx):
        self.n_cluster += 1
        self.W[idx] = fuzzy_and(x, self.W[idx])
        # 创建新的权重矩阵
        new_weight = torch.ones(self.n_cluster, self.input_dim, dtype=torch.float32, device=self.device)

        # 保留原有权重
        new_weight[:-1, :] = self.W.data

        # 更新权重参数 (保持不需要梯度)
        self.W = new_weight



if __name__ == '__main__':
    artnet = FuzzyART(1, 1,1, 1, 1, 'cpu')
    artnet.init_fuzzy_art_net(3)
    print(artnet.W)
    print(artnet.calculate_choice_function(torch.tensor([0.2, 0.2, 0.2], device='cpu')))
    print(artnet.judge_match(torch.tensor([2, 0.2, 0.2], device='cpu')))