"""
Generating algorithmic dataset for the experiments.
"""

# Standard libraries
from typing import Callable

# Third-party dependencies
import torch
from torch import Tensor


def create_algorithmic(size: int, combine: Callable[[Tensor, Tensor], Tensor],
                       device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.int32) -> Tensor:
    """
    Creates a table of size (size x size) by applying combine to two matrices, where one of them
    represents the first, the other the second index.
    :param size: The size of the table.
    :param combine: The function to produce (a o b) where o is a binary operator.
    :param device: Device to put the tensor on.
    :param dtype: Type of the tensor.
    :return: Returns the data as a tensor of shape (size*size, 3), for each row
    the first element represents x, the second y, the third (x o y) for a binary operator o.
    """
    a = torch.arange(size, device=device, dtype=dtype)[..., None].repeat((1, size))
    b = torch.arange(size, device=device, dtype=dtype).repeat((size, 1))

    combined = combine(a, b)
    res = torch.stack([a, b, combined]).permute((1, 2, 0)).view((-1, 3))
    return res


def multiply(p: int) -> Callable[[Tensor, Tensor], Tensor]:
    """
    x * y (mod p) from paper.
    Returns a combinator function for multiplication.
    :param p: The size of the table.
    :return: The combinator function.
    """
    return lambda x, y: (x * y) % p


def divide(p: int) -> Callable[[Tensor, Tensor], Tensor]:
    """
    x / y (mod p) from paper, for 0 <= x < p, 0 < y < p
    Returns a combinator function for division.
    :param p: The size of the table.
    :return: The combinator function.
    """
    def f(x, y):
        mp = multiply(p)(x, y)
        res = torch.zeros_like(x)
        # Fill array in a loop.
        for a in range(res.shape[0]):
            for b in range(1, res.shape[1]):
                res[a, b] = (mp[b] == a).nonzero()[0, 0]
        return res
    return f


out = create_algorithmic(97, divide(97), torch.device("cuda:0"))
print(out.shape)

# out = create_algorithmic(5, lambda x, y: ((x+5).div(y+5, rounding_mode='trunc')) % 5, torch.device("cuda:0"))
# print(out)
