from numpy.random import randint, random, choice
import numpy as np

EMPTY       = 0
CANCER_CELL = 1
TKILLER_CELL = 2


from typing import List, Tuple, Set

class CancerImmuneModel:
    """2D spatial model of a tissue sample with cancerous growth

    Properties:
        dim        (Tuple[int, int]): Length and width of the model
        
        cancerLattice (ndarray[int]): Lattice containing the cancer cells of the system
        immuneLattice (ndarray[int]): Lattice containing the immune cells of the system
        
        pImmuneKill          (float): Probability that an immune cell kills a cancer cell
        pCancerMult          (float): Probability that a cancer cell multiplies during a timestep

        cancerCells    (Set[Tuple[int, int]]): Set of all cell coordinates containing cancer cells, used for scheduling
        immuneCells    (Set[Tuple[int, int]]): List of all cell coordinates containing immune cells, used for scheduling
        cancerCells_t1 (Set[Tuple[int, int]]): Set of all cell coordinates containing cancer cells fopr next timestep
        immuneCells_t1 (Set[Tuple[int, int]]): List of all cell coordinates containing immune cells fopr next timestep
    """
    def __init__(self, length: int, width: int, pImmuneKill = 1.0, pCancerMult = 0.05) -> None:
        """
        initializer function

        Args:
            length: Length of the lattice
            width : Width of the lattice
        """
        self.time = 0

        self.dim =  (length, width)
        
        self.cancerLattice = np.zeros((length, width), dtype=int)
        self.immuneLattice = np.zeros((length, width), dtype=int)

        self.pImmuneKill   = pImmuneKill
        self.pCancerMult   = pCancerMult

        self.cancerCells   :  Set[Tuple[int, int]] = set()
        self.immuneCells   :  Set[Tuple[int, int]] = set()
        self.cancerCells_t1:  Set[Tuple[int, int]] = set()
        self.immuneCells_t1:  Set[Tuple[int, int]] = set()
    
    def get_nCancerCells(self) -> int:
        """Returns the amount of cancer cells in the system"""
        return len(self.cancerCells)
    
    def get_nImmuneCells(self) -> int:
        """Returns the amount of immune cells in the system"""
        return len(self.immuneCells)

    def seedCancer(self, nCells: int) -> None:
        """
        Places nCells cancer cells on random cells in the cancer lattice
        """
        for i in range(nCells):
            row = randint(0, self.dim[0] - 1)
            col = randint(0, self.dim[1] - 1)

            self._addCancer((row, col))
        
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

            if self.cancerLattice[row, col] == 0:
                self._addImmune((row, col))
                cells += 1
        
        return cycles

    def _neighborlist(self, cell: Tuple[int, int], periodic=False, includeSelf=False, 
                      lattice: np.ndarray[int] = None, emptyOnly=True) -> List[Tuple[int, int]]:
        """
        Returns a list of neighboring cell coordinates for a given cell using Moore's neighborhood.

        Args:
            cell    (Tuple[int, int]): The coordinates of the cell
            periodic           (bool): Wether periodic boundary conditions are enforced
            includeSelf        (bool): Should own cell be included in neighborhood
            lattice (np.ndarray[int]): The lattice to be used to check for occupancy
            emptyOnly          (bool): If a lattice has been specified, only return empty cells if true
                                       or occupied cells if false.

        Returns (List[Tuple[int, int]]: A list of valid neighbor cells coordinates for the cell.
        """
        output = []
        row, col = cell

        for vertical in [-1, 0, 1]: 
        # Loop over surrounding rows
            for horizontal in [-1, 0, 1]: 
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

    def propagateCancerCell(self, cell: Tuple[int, int]) -> int:
        """
        Propagates a cancer cell by multiplying. Adds new cancer cells to self.cancerLattice and to
        cancerCells_t1.

        Args:
            Cell (Tuple[int, int]): Cell coordinate
        
        Returns (int): 0 if not able to multiply by chance, 1 if not able to multiply from overcrowding
                       and 2 if sucessfully multiplied
        """
        self._addCancer(cell)

        if random() > self.pCancerMult:
            # Stop if cell will not multiply
            return 0
        
        neighbors = self._neighborlist(cell, lattice=self.cancerLattice)

        if not neighbors:
            # Stop multiplication attempt if there are no free spaces in neighborhood
            return 1
        
        targetID = 0
        if len(neighbors) > 1:
            # Take only position if single available space
            targetID = randint(0, len(neighbors) - 1)

        newCell = neighbors[targetID]

        self._addCancer(newCell)
        return 2
    
    def multiTKiller(self, cell):
        if self.propagateTKiller(cell) != 1:
            return 0
        
        freeSpace = self._neighborlist(cell, lattice=self.cancerLattice)

        if not freeSpace:
            self.seedImmune(1)
            return 1
  
        targetID = 0
        if len(freeSpace) > 1:
            # Take only position if single available space
            targetID = randint(0, len(freeSpace) - 1)

        newCell = freeSpace[targetID]

        self._addImmune(newCell)
        return 2
    
    def deleteTkiller(self, cell):        
        # Check for cancer cells in the neighborhood.
        cancer_neighbors = 0
        tkiller_neighbors = 0
        neighbors = self._neighborlist(cell, lattice=self.cancerLattice, emptyOnly=False)
        for neighbor in neighbors:
            if self.cancerLattice[neighbor[0], neighbor[1]] == CANCER_CELL:
                cancer_neighbors += 1
            if self.immuneLattice[neighbor[0], neighbor[1]] == TKILLER_CELL:
                tkiller_neighbors += 1
        
        # If there are cancer cells nearby, do nothing.
        if cancer_neighbors > 0:
            return 0

        # If there are no cancer cells but more than 2 T-Killer cells, the cell dies in the next timestep.
        if tkiller_neighbors > 5:
            # Remove the current cell from the next timestep's set of T-Killer cells.
            self._removeImmune(cell)
            return 1

        return 0
    
    def propagateTKiller(self, cell) -> int:
        """
        Propagates a T-Killer immune cell. Moves TKILLER_CELL on immuneLattice and adds new position
        to immunecells_t1. Cell will not move if sucessfully killing a cancer cell

        Args:
            Cell (Tuple[int, int]): Cell coordinate

        Returns (int): 0 if unable to move from overcrowding, 1 if succesfull in killing a cancer cell 
                       and 2 if moving without killing a cancer cell.
        """
        self._addImmune(cell)
        if self.cancerLattice[cell[0], cell[1]] and random() <= self.pImmuneKill:
        # If currently occupying a cell with a cancer cell, randomly kill it. If sucessful, stay, else continue
            self._removeCancer(cell)
            return 1
    
        moves = self._neighborlist(cell, True, lattice=self.immuneLattice)

        if not moves:
        # If no possible moves are available, stay, else continue.
            return 0

        moveIndex = 0
        if len(moves) > 1:
        # If more than 1 move is possible, pick a random option
            moveIndex = randint(0, len(moves) - 1)
        
        target = moves[moveIndex]
        
        # Move cell from current location to target location
        self._removeImmune(cell)
        self._addImmune(target) # Add target location to schedule

        self.deleteTkiller(target)
        return 2

    def _addCancer(self, cell: Tuple[int, int]):
        self.cancerLattice[cell[0], cell[1]] = CANCER_CELL # Create new cell on lattice 
        self.cancerCells_t1.add(cell) # add new cell to schedule

    def _removeCancer(self, cell: Tuple[int, int]):
        if cell not in self.cancerCells_t1:
            print(f"Warning: Specified cancer cell {cell[0]},{cell[1]} not in t1 scheduler")
        self.cancerLattice[cell[0], cell[1]] = EMPTY # Create new cell on lattice 
        self.cancerCells_t1.remove(cell) # remove cell from scheduler

    def _addImmune(self, cell: Tuple[int, int]):
        self.immuneLattice[cell[0], cell[1]] = TKILLER_CELL
        self.immuneCells_t1.add(cell)

    def _removeImmune(self, cell: Tuple[int, int]):
        if cell not in self.immuneCells_t1:
            print(f"Warning: Specified cancer cell {cell[0]},{cell[1]} not in t1 scheduler")
        self.immuneLattice[cell[0], cell[1]] = EMPTY
        self.immuneCells_t1.remove(cell)
    
    def timestep(self):
        """
        Propagates the model by 1 step
        """ 

        for cell in self.cancerCells:
            self.propagateCancerCell(cell)
        for cell in self.immuneCells:
            self.multiTKiller(cell)
        
        if self.time % 1000:
                self.seedCancer(1)

        
        self.cancerCells = self.cancerCells_t1
        self.cancerCells_t1 = set()
        self.immuneCells = self.immuneCells_t1
        self.immuneCells_t1 = set()

        self.time += 1