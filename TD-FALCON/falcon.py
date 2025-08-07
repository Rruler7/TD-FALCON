import torch
import random
from torch import nn
import numpy as np
from algorithm.fuzzy_art import FuzzyART
from algorithm.fusion_art import FusionART
from algorithm.art2 import ART2
from typing import Optional, Literal, Tuple, Union, List
from algorithm.utils import fuzzy_and, comlement_code

""""""
class FALCON:
    def __init__(
            self,
            state_art: FuzzyART,
            action_art: FuzzyART,
            reward_art: FuzzyART,
            channel_dims: Union[List[int], np.ndarray] = list[int],
            td_alpha: float = 1.0,
            td_lambda: float = 1.0,
            optimality: str = 'max',
            device: str = 'cpu'
    ):
        self.fusion_art = FusionART(
            modules=[state_art, action_art, reward_art],
            channel_dims=channel_dims,
            device=device
        )

        self.td_alpha = td_alpha
        self.td_lambda = td_lambda

        self.optimality = optimality

    def random_action(self):
        """
        随机动作
        :return: 返回动作（array）， Q
        """
        dim = self.fusion_art.channel_dims[1]
        action = np.zeros(dim, dtype=int)
        # 随机选择一个索引（0到dim-1之间）
        idx = np.random.randint(0, dim)
        # 将该索引位置设为1
        action[idx] = 1
        Q = comlement_code(0.5, 'fuzzy')
        return action, Q


    def get_action(self, S):
        """
        利用
        :param s: 输入向量(array)
        :return: 动作（array），Q(array)
        """
        A = [1, 1, 1, 1]
        if self.optimality == 'max':    # 选取最大奖励值
            Q = [1, 0]
        else:
            Q = [0, 1]
        x = torch.tensor(np.hstack([S, A, Q]), dtype=torch.float32, device='cuda')  # 输入向量
        idx = self.fusion_art.resonance_search(x, "rhos", self.optimality)
        if self.fusion_art.committed[idx] == True:
            action = self.fusion_art.get_x_fuzzy_and_w(x, idx, 1)
            q = self.fusion_art.get_w(idx, 2)
            return action, q
        else:
            return self.random_action()

    def learn(self, S, A, Q, r, NS):
        # 寻找最大或小Q值
        AN, Qmax = self.get_action(NS)

        # 解码
        q = Q[0] / (Q[0] + Q[1])
        qmax = Qmax[0] / (Qmax[0] + Qmax[1])

        Qnew = q + self.td_alpha * (r + self.td_lambda * qmax - q) * (1 - q)
        # Qnew = Q[0] + self.td_alpha * (r + self.td_lambda * QN - Q[0])
        Qnew = comlement_code(Qnew, 'fuzzy')

        x = np.hstack([S, A, Qnew])
        x = torch.tensor(x, device='cuda', dtype=torch.float32)
        best_match_idx = self.fusion_art.resonance_search(x, "rhol", self.optimality)
        self.fusion_art.updatew(x, best_match_idx)




if __name__ == "__main__":
    art_state = FuzzyART(rhos=0.0, rhol=0.0, alpha=0.1, beta=1.0, gama=0.33, device='cuda')
    art_action = FuzzyART(rhos=0.0, rhol=1.0, alpha=0.1, gama=0.33, beta=1.0, device='cuda')
    art_reward = FuzzyART(rhos=0.5, rhol=0.75, alpha=0.1, gama=0.33, beta=1.0, device='cuda')
    cls = FALCON(art_state, art_action, art_reward, [2, 2, 2], 0.5, 0.1, 'cuda')

