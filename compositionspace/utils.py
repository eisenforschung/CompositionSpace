"""Utility functions."""

import numpy as np


def ceil_to_multiple(number, multiple):
    return multiple * np.ceil(number / multiple)


def floor_to_multiple(number, multiple):
    return multiple * np.floor(number / multiple)

# ceil to a multiple of 1.5
# print(ceil_to_multiple(23.0000000000000000000000000000000000000, 1.5))
# floor to a multiple of 1.5
# print(floor_to_multiple(-23.0000000000000000000000000000000000000, 1.5))
