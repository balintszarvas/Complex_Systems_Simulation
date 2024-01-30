from numpy.random import randint, random, choice
from typing import List, Tuple


def randomTuple(options: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Returns a random Tuple from a lis of tuples. Added as a replacement for numpy.random.choice as
    it strictly works for 1D lists and regards the content from a Tuple as a seperate dimension.
    """
    choiceID = 0
    if len(options) > 1:
        # Take only position if single available space
        choiceID = randint(0, len(options) - 1)

    return options[choiceID]

def mooreMaxSize(radius: int, includeSelf = False) -> int:
    return (2 * radius + 1)**2 - int(includeSelf)
