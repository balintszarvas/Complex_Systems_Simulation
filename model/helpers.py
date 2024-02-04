from numpy.random import randint
from typing import List, Tuple

"""Helper function for bacteriaImmune model"""

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

def mooreMaxSize(radius: int, dontIncludeSelf = False) -> int:
    """
    Returns the amount of tiles in a moore neighborhood of a given radius, either inclusive or not inclusive of the
    center tile.
    """
    return (2 * radius + 1)**2 - int(dontIncludeSelf)
