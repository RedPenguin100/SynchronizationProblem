from typing import List
import math


def average(l: List):
    if len(l) == 0:
        return math.nan
    return sum(l) / len(l)
