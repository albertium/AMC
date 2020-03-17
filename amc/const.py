
from enum import Enum


class Bound(Enum):
    UPPER = 'ub'
    LOWER = 'lb'


class Op(Enum):
    ADD = 0
    MINUS = 1
