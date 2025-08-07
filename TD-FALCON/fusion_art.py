import numpy as np
import torch

from algorithm.utils import get_channel_position_tuples
from algorithm.fuzzy_art import FuzzyART
from typing import Optional, Union, Callable, List, Literal, Tuple, Dict
from algorithm.utils import fuzzy_and


class FusionART:
    def __init__(
            self,
            modules: List[FuzzyART],
            channel_dims: Union[List[int], np.ndarray],
            device: str
    ):
        self.modules = modules
        self.n = len(self.modules)
        self.channel_dims = channel_dims
        self._channel_indices = get_channel_position_tuples(self.channel_dims)
        self.dim_ = sum(channel_dims)
        self.committed = [False]
        self.n_cluster = len(self.committed)
        self.device = device
        self.init_fusion_art_net()

    def init_fusion_art_net(self):
        for module, i in zip(self.modules, self.channel_dims):
            module.init_fuzzy_art_net(i)

    def calculate_choice_function(self, x):
        """
        x: 输入向量
        return: 激活函数值
        """
        Tj = torch.zeros(self.n_cluster, dtype=torch.float32, device=self.device)
        for module, position in zip(self.modules, self._channel_indices):
            Tjj = module.calculate_choice_function(x[position[0]:position[1]])
            Tj = Tj + Tjj

        return Tj

    def judge_match(self, x, rho):
        match = np.array([True for i in range(self.n_cluster)])
        for module, position in zip(self.modules, self._channel_indices):
            is_match = np.array(module.judge_match(x[position[0]: position[1]], rho))
            match = match & is_match

        return match

    def resonance_search(self, x, rho, optimality):
        """
        :param x:输入
        :return: 匹配节点下标
        """
        # 转换为NumPy数组
        Tj = self.calculate_choice_function(x).cpu().numpy()
        match = self.judge_match(x, rho)

        # 筛选出布尔值为True的元素及其索引
        mask = match
        valid_values = Tj[mask]
        valid_indices = np.where(mask)[0]  # 获取True对应的索引

        # 找到最大值在有效元素中的位置，再映射到原始索引

        max_pos = np.argmax(valid_values)
        min_pos = np.argmin(valid_values)
        if optimality == 'max':
            return valid_indices[max_pos]
        else:
            return valid_indices[min_pos]

    def get_x_fuzzy_and_w(self, x, idx, channel):
        """
        x fuzzy and 某个节点对应的w
        :param x:
        :param idx:
        :return: array
        """
        xstart = self._channel_indices[channel][0]
        xend = self._channel_indices[channel][1]
        x = x[xstart:xend]
        w = self.modules[channel].W[idx]
        return fuzzy_and(x, w).cpu().numpy()

    def get_w(self, idx, channel):
        """
        获得权重
        :param idx:
        :param channel:
        :return: 权重，tensor
        """
        w = self.modules[channel].W[idx]
        return w.cpu().numpy()

    def updatew(self, x, idx):
        if self.committed[idx]:
            a_start = self._channel_indices[1][0]
            a_end = self._channel_indices[1][1]
            x[a_start: a_end] = fuzzy_and(x[a_start: a_end], self.modules[1].W[idx])
            for (module, position) in zip(self.modules, self._channel_indices):
                module.updatew(x[position[0]:position[1]], idx)
        else:
            for (module, position) in zip(self.modules, self._channel_indices):
                module.add_cluster(x[position[0]:position[1]], idx)
            self.n_cluster += 1
            self.committed[idx] = True
            self.committed.append(False)


