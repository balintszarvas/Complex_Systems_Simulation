import numpy.random
from numpy.random import randint, random, choice
from ..helpers import randomTuple, mooreMaxSize
import numpy as np

numpy.random.seed(10)
EMPTY       = 0
bacteria_CELL = 1
TKILLER_CELL = 2


from typing import List, Tuple, Set

class bacteriaImmuneModel:
    """2D spatial model of a tissue sample with bacteriaous growth

    Properties:
        dim        (Tuple[int, int]): Length and width of the model

        bacteriaLattice (ndarray[int]): Lattice containing the bacteria cells of the system
        immuneLattice   (ndarray[int]): Lattice containing the immune cells of the system

        pImmuneKill            (float): Probability that an immune cell kills a bacteria cell
        pBacteriaMult          (float): Probability that a bacteria cell multiplies during a timestep

        bacteriaCells    (Set[Tuple[int, int]]): Set of all cell coordinates containing bacteria cells.
                                                 Used for scheduling.
        immuneCells      (Set[Tuple[int, int]]): List of all cell coordinates containing immune cells.
                                                 Used for scheduling.
        bacteriaCells_t1 (Set[Tuple[int, int]]): Set of all cell coordinates containing bacteria cells for next timestep
        immuneCells_t1   (Set[Tuple[int, int]]): List of all cell coordinates containing immune cells fopr next timestep

    Quick Implementation Guide
    If one wants to implement the model into their own project as is, the following steps should be taken.
        1. Initialize a new `BacteriaImmuneModel` object.
        2. (optional) Seed the model with an initial population of bacterial cells for the desired amound
        of cells nCells using `BacteriaImmuneModel.seedBacteria(nCells: int)`
        3. Seed the model with an initial population of immune cells for the desired amount of cells
           nCells using `BacteriaImmuneModel.seedImmune(nCells: int)`
        4. Call `BacteriaImmuneModel.timestep()` to update the lattice. The current populations of immunecells
           and bacterial cells can be obtained using `BacteriaImmuneModel.get_nImmuneCells()` and
           `BacteriaImmuneModel.get_nbacteriaCells()` respectively
    """
    def __init__(self, length: int, width: int, pImmuneKill = 1.0, pBacteriaMult = 0.05, pBacteriaSpawn = 0.01) -> None:
        """
        initializer function

        Args:
            length           (int): Length of the lattice.
            width            (int): Width of the lattice.
            pImmuneKill    (float): Chance an immune cell kills a bacteria cell.
            pBacteriaMult  (float): Chance a bacteria cell multiplies every timestep.
            pBacteriaSpawn (float): Chance a normal cell becomes bacteriaous per timestep.
        """
        self.time = 0

        self.dim =  (length, width)

        self.bacteriaLattice = np.zeros((length, width), dtype=int)
        self.immuneLattice = np.zeros((length, width), dtype=int)

        self.pImmuneKill   = pImmuneKill
        self.pBacteriaMult   = pBacteriaMult
        self.pBacteriaSpawn = pBacteriaSpawn

        self.bacteriaCells   :  Set[Tuple[int, int]] = set()
        self.immuneCells   :  Set[Tuple[int, int]] = set()
        self.bacteriaCells_t1:  Set[Tuple[int, int]] = set()
        self.immuneCells_t1:  Set[Tuple[int, int]] = set()

    def get_nbacteriaCells(self) -> int:
        """Returns the amount of bacteria cells in the system"""
        return len(self.bacteriaCells)

    def get_nImmuneCells(self) -> int:
        """Returns the amount of immune cells in the system"""
        return len(self.immuneCells)

    def seedbacteria(self, nCells: int) -> None:
        """Places nCells bacteria cells on random cells in the bacteria lattice"""
        for i in range(nCells):
            row = randint(0, self.dim[0] - 1)
            col = randint(0, self.dim[1] - 1)

            self._addbacteria((row, col))

    def seedImmune(self, nCells: int) -> int:
        """
        Places nCells immune cells on random cells in the immune lattice

        Returns the amount of placement attempts made
        """
        cells = 0
        cycles = 0
        while cells < nCells:
            cycles += 1
            row = randint(0, self.dim[0] - 1)
            col = randint(0, self.dim[1] - 1)

            if self.bacteriaLattice[row, col] == 0:
                self._addImmune((row, col))
                cells += 1

        return cycles

    def _neighborlist(self, cell: Tuple[int, int], periodic=False, includeSelf=False,
                      lattice: np.ndarray = None, emptyOnly=True, radius=1) -> List[Tuple[int, int]]:
        """
        Returns a list of neighboring cell coordinates for a given cell using Moore's neighborhood.

        Args:
            cell    (Tuple[int, int]): The coordinates of the cell
            periodic           (bool): Wether periodic boundary conditions are enforced
            includeSelf        (bool): Should own cell be included in neighborhood
            lattice (np.ndarray[int]): The lattice to be used to check for occupancy
            emptyOnly          (bool): If a lattice has been specified, only return empty cells if true
                                       or occupied cells if false.
            radius              (int): Radius for the moore neighborhood

        Returns (List[Tuple[int, int]]: A list of valid neighbor cells coordinates for the cell.
        """
        output = []
        row, col = cell

        for vertical in range(-radius, radius + 1):
        # Loop over surrounding rows
            for horizontal in range(-radius, radius + 1):
            # Loop over surrounding collumns
                if vertical == 0 and horizontal == 0 and not includeSelf:
                # Skip neighbor if its the same cell unless specifically included
                    continue

                neighborRow = row + vertical
                neighborCol = col + horizontal

                if periodic:
                # If periodic boundary conditions are used, take the modulo of the coordinate and the dimensions
                    neighborRow = neighborRow % self.dim[0]
                    neighborCol = neighborCol % self.dim[1]

                if (neighborRow >= 0 and neighborRow < self.dim[0] and neighborCol >= 0 and neighborCol < self.dim[1]):
                # Exclude all out of bounds cells
                    if not lattice is None and emptyOnly == bool(lattice[neighborRow, neighborCol]):
                    # when a lattice is specified, skip cell if emptyonly and a cell is occupied or
                    # if not emptyonly and a cell is not occupied (XNOR)
                        continue

                    output.append((neighborRow, neighborCol))
        return output

    def propagatebacteriaCell(self, cell: Tuple[int, int]) -> int:
        """
        Propagates a bacteria cell by multiplying. Adds new bacteria cells to self.bacteriaLattice and to
        bacteriaCells_t1.

        Args:
            Cell (Tuple[int, int]): Cell coordinate

        Returns (int): 0 if not able to multiply by chance, 1 if not able to multiply from overcrowding
                       and 2 if sucessfully multiplied
        """
        self._addbacteria(cell)

        if random() > self.pBacteriaMult:
            # Stop if cell will not multiply
            return 0

        neighbors = self._neighborlist(cell, lattice=self.bacteriaLattice, periodic=True)

        if not neighbors:
            # Stop multiplication attempt if there are no free spaces in neighborhood
            return 1

        newCell = randomTuple(neighbors)

        self._addbacteria(newCell)
        return 2

    def multiTKiller(self, cell: Tuple[int, int]):
        """
        Multiply a TKiller cell by placing a new cell in it's direct neighborhood, but only if there
        are nearby bacteria cells.

        Args:
            Cell (Tuple[int, int]): The coordinates of the cell attempting to multiply
        """
        freeSpace = self._neighborlist(cell, lattice=self.bacteriaLattice)

        if not freeSpace:
            # self.seedImmune(1)
            return 1

        newCell = randomTuple(freeSpace)
        self._addImmune(newCell)
        return 2

    def deleteTkiller(self, cell: Tuple[int, int]):
        """
        Check if a T-killer Cell is to be removed, based on the presence of bacteria cells nearby and
        immune cell crowding

        Args:
            Cell (Tuple[int, int]): The coordinates of the cell attempting to multiply
            cDetectionRadius (int): Radius for the moore neighborhood where to detect bacteria cells.
        """
        C_DETECTION_RADIUS = 1   # Detection radius for bacteria cells
        I_DETECTION_RADIUS = 1   # Detection radius for Immune cells
        I_DENS_LIMIT       = 1/8 # Density limit for immune cells

        P_RDEATH = 0.01          # Probability of a cell randomly dying

        # Check for bacteria cells in the neighborhood.
        bacteria_neighbors = self._neighborlist(cell, lattice=self.bacteriaLattice, emptyOnly=False, radius=C_DETECTION_RADIUS)

        # If there are bacteria cells nearby, do nothing.
        if len(bacteria_neighbors) > 0:
            return 0

        tkiller_neighbors = self._neighborlist(cell, lattice=self.immuneLattice, emptyOnly=False, radius=I_DETECTION_RADIUS)

        # If there are no bacteria cells but more than 2 T-Killer cells, the cell dies before the next timestep.
        if len(tkiller_neighbors) / mooreMaxSize(I_DETECTION_RADIUS) > I_DENS_LIMIT:
            # Remove the current cell from the next timestep's set of T-Killer cells.
            self._removeImmune(cell)
            return 1

        if self.get_nbacteriaCells() != 0:
            return 0

        if random() <= P_RDEATH:
            self._removeImmune
            return 1

        return 2

    def propagateTKiller(self, cell) -> int:
        """
        Propagates a T-Killer immune cell. Moves TKILLER_CELL on immuneLattice and adds new position
        to immunecells_t1. Cell willn ot move if sucessfully killing a bacteria cell

        Args:
            Cell (Tuple[int, int]): Cell coordinate

        Returns (int): 0 if unable to move from overcrowding, 1 if succesfull in killing a bacteria cell
                       and 2 if moving without killing a bacteria cell.
        """
        self._addImmune(cell)

        if self.bacteriaLattice[cell[0], cell[1]] and random() <= self.pImmuneKill:
        # If currently occupying a cell with a bacteria cell, randomly kill it
            self._removebacteria(cell)
            self.multiTKiller(cell) # Attempt to multiply after killing a cell
            return 1

        moves = self._neighborlist(cell, True, lattice=self.immuneLattice)

        if not moves:
        # If no possible moves are available, stay, else continue.
            return 0

        target = randomTuple(moves)

        # Move cell from current location to target location
        self._removeImmune(cell)
        self._addImmune(target)

        self.deleteTkiller(target)
        return 2

    def _addbacteria(self, cell: Tuple[int, int]):
        """Places cell on the bacteria lattice and adds it to the future scheduler"""
        self.bacteriaLattice[cell[0], cell[1]] = bacteria_CELL # Create new cell on lattice
        self.bacteriaCells_t1.add(cell) # add new cell to schedule

    def _removebacteria(self, cell: Tuple[int, int]):
        """Removes a cell from the bacteria lattice and removes it from the future scheduler"""
        if cell not in self.bacteriaCells_t1:
            print(f"Warning: Specified bacteria cell {cell[0]},{cell[1]} not in t1 scheduler")
        self.bacteriaLattice[cell[0], cell[1]] = EMPTY # remove cell from lattice
        self.bacteriaCells_t1.remove(cell) # remove cell from scheduler

    def _addImmune(self, cell: Tuple[int, int]):
        """Places cell on the immune lattice and adds it to the future scheduler"""
        self.immuneLattice[cell[0], cell[1]] = TKILLER_CELL # Create new cell on lattice
        self.immuneCells_t1.add(cell) # add new cell to schedule

    def _removeImmune(self, cell: Tuple[int, int]):
        """Removes a cell from the immune lattice and removes it from the future scheduler"""
        if cell not in self.immuneCells_t1:
            print(f"Warning: Specified bacteria cell {cell[0]},{cell[1]} not in t1 scheduler")
        self.immuneLattice[cell[0], cell[1]] = EMPTY # remove cell from lattice
        self.immuneCells_t1.remove(cell) # remove cell from scheduler

    def timestep(self):
        """Propagates the model by 1 timestep"""

        for cell in self.bacteriaCells:
            self.propagatebacteriaCell(cell)
        for cell in self.immuneCells:
            self.propagateTKiller(cell)

        if random() < self.pBacteriaSpawn:
            self.seedbacteria(1)

        self.bacteriaCells = self.bacteriaCells_t1
        self.bacteriaCells_t1 = set()
        self.immuneCells = self.immuneCells_t1
        self.immuneCells_t1 = set()

        self.time += 1
