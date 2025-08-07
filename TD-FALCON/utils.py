import torch
import numpy as np
from typing import Optional, Union, Callable, List, Literal, Tuple, Dict

def get_channel_position_tuples(
    channel_dims: List[int],
) -> List[Tuple[int, int]]:
    """Generate the start and end positions for each channel in the input data.

    Parameters
    ----------
    channel_dims : list of int
        A list representing the number of dimensions for each channel.

    Returns
    -------
    list of tuple of int
        A list of tuples where each tuple represents the start and end index for a
        channel.

    """
    positions = []
    start = 0
    for length in channel_dims:
        end = start + length
        positions.append((start, end))
        start = end
    return positions

def fuzzy_and(x, w):
    """模糊AND操作 (元素最小值)"""
    return torch.minimum(x, w)

def comlement_code(x, type):
    if type == 'fuzzy':
        return np.array([x, 1.0 - x])
    elif type == 'art2':
        return np.array([x, np.sqrt(1 - np.square(x))])

if __name__ == '__main__':
    x = 0.5
    print(comlement_code(x, 'art2'))
