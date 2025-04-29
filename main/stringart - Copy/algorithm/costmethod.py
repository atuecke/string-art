from enum import Enum, auto

class CostMethod(Enum):
    DIFFERENCE = auto()
    MEAN = auto()
    SMSE = auto()
    SMRSE = auto()
    MEDIAN = auto()