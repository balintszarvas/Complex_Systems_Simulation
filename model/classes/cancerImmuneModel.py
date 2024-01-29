from numpy.random import randint, random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import powerlaw
from typing import List, Tuple, Set
EMPTY       = 0
CANCER_CELL = 1
TKILLER_CELL = 2



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
    def __init__(self, length: int, width: int, pImmuneKill = 0.8, pCancerMult = 0.5, pCancerSpawn=0.5, attack_range = 2) -> None:
        """
        initializer function

        Args:
            length: Length of the lattice
            width : Width of the lattice
        """
        self.time = 0
        self.attack_range = attack_range

        self.dim =  (length, width)

        self.cancerLattice = np.zeros((length, width), dtype=int)
        self.immuneLattice = np.zeros((length, width), dtype=int)

        self.pImmuneKill   = pImmuneKill
        self.pCancerMult   = pCancerMult

        self.cancerCells   :  Set[Tuple[int, int]] = set()
        self.immuneCells   :  Set[Tuple[int, int]] = set()
        self.cancerCells_t1:  Set[Tuple[int, int]] = set()
        self.immuneCells_t1:  Set[Tuple[int, int]] = set()

        self.immune_cells_over_time =[]

        self.cluster_sizes = []
        self.cluster_durations = defaultdict(int)

        self.pCancerEmergence = pCancerSpawn

    def detect_clusters(self):
        visited = set()
        clusters = []

        def dfs(cell):
            if self.cancerLattice[cell[0], cell[1]] != CANCER_CELL:
                return 0
            stack = [cell]
            size = 0
            while stack:
                current_cell = stack.pop()
                if current_cell in visited:
                    continue
                visited.add(current_cell)
                size += 1
                for neighbor in self._neighborlist(current_cell):
                    if neighbor not in visited and self.cancerLattice[neighbor[0], neighbor[1]] == CANCER_CELL:
                        stack.append(neighbor)
            return size

        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                if (row, col) not in visited and self.cancerLattice[row, col] == CANCER_CELL:
                    clusters.append(dfs((row, col)))
        return clusters


    def plot_cluster_sizes(self):
        # Fit the power law to the raw cluster sizes
        power_results = powerlaw.Fit(self.cluster_sizes)
        R, p = power_results.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Plot the PDF using the powerlaw results on the created axis
        power_results.plot_pdf(ax=ax, color='b', linestyle='-', label='Power law fit')
        power_results.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label='Fitted power law')

        # Set labels and title using the axis methods
        ax.set_xlabel('Log of Cluster Size')
        ax.set_ylabel('Log of Frequency')
        ax.set_title('Log-Log Plot of Cancer Cell Cluster Sizes')

        # Add text annotation on the plot
        ax.text(0.3, 0.95, f'Power law vs lognormal={R:.2f} \np-value of fit={p:.2f}', 
            horizontalalignment='center', verticalalignment='top', 
            transform=ax.transAxes, bbox={'facecolor': 'white', 'alpha': 0.5})

        # Add legend and show the plot
        ax.legend()
        plt.savefig('cluster_sizes.png')


    def cancer_emergence(self):
        """
        Simulate the random emergence of new cancer cells by selecting a random empty cell
        """
        if np.random.random() <self.pCancerEmergence:
            empty_cells=[(row,col) for row in range(self.dim[0])
                           for col in range(self.dim[1]) if self.cancerLattice[row,col] == EMPTY]
            if empty_cells:
                new_cancer_cell =random.choice(empty_cells)
                self._addCancer(new_cancer_cell)

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

        if random.random() > self.pCancerMult:
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
        """
        Creates a new T-Killer cell in a neighboring free space, if the T-Killer cell successfully kills a cancer cell.
        If no free space is available, it seeds a new immune cell randomly.

        Args:
            cell (Tuple[int, int]): Coordinates of the T-Killer cell.

        Returns:
            int: 0 if the cell hasn't killed a cancer cell, 1 if no free space is available for new cell creation,
                and 2 if a new T-Killer cell is created in the neighborhood.
        """
        if self.propagateTKiller(cell) != 1:
            return 0

        freeSpace = self._neighborlist(cell, lattice=self.cancerLattice)

        if not freeSpace:
            # self.seedImmune(1)
            return 1

        targetID = 0
        if len(freeSpace) > 1:
            # Take only position if single available space
            targetID = randint(0, len(freeSpace) - 1)

        newCell = freeSpace[targetID]

        self._addImmune(newCell)
        return 2

    def deleteTkiller(self, cell):
        """
        Deletes a T-Killer cell if there are no cancer cells in the neighborhood and a random chance condition is met.
        The T-Killer cell is not deleted if there are any neighboring cancer cells. 
        If there are no neighboring cancer cells and the simulation still has cancer cells,
        the T-Killer cell is not deleted. Only if no cancer cells are nearby and the chance condition is met,
        the T-Killer cell is marked for deletion.

        Args:
            cell (Tuple[int, int]): Coordinates of the T-Killer cell.

        Returns:
            int: 0 if the T-Killer cell is not deleted (due to presence of cancer cells or ongoing cancer),
                1 if the cell is marked for deletion based on the random chance condition,
                2 as a default return when none of the conditions are met.
        """
        # Check for cancer cells in the neighborhood.
        cancer_neighbors = self._neighborlist(cell, lattice=self.cancerLattice, emptyOnly=False)
        tkiller_neighbors = self._neighborlist(cell, lattice=self.immuneLattice, emptyOnly=False)

        # If there are cancer cells nearby, do nothing.
        if len(cancer_neighbors) > 0:
            return 0

        if self.get_nCancerCells() != 0:
            return 0

        if random.random() <= 0.01:
            self._removeImmune
            return 1

        return 2

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
        if self.cancerLattice[cell[0], cell[1]] and random.random() <= self.pImmuneKill:
            self._removeCancer(cell)
            return 1

        # check for cancer within atttack_range
        cancer_in_range =self.get_cells_in_range(cell, self.attack_range)
        cancer_in_range =[c for c in cancer_in_range if self.cancerLattice[c[0],c[1]] ==CANCER_CELL]

        if cancer_in_range:
            target =min(cancer_in_range, key=lambda c: self.distance(cell, c))
            move =self.get_step_towards(cell, target)
        else:
            # random
            moves =self._neighborlist(cell,True,lattice=self.immuneLattice)
            if not moves:
                return 0
            move = random.choice(moves)

        self._removeImmune(cell)
        self._addImmune(move)
        return 2

    def distance(self, cell1, cell2):
        return np.sqrt((cell1[0] -cell2[0])** 2 +(cell1[1] - cell2[1]) **2 )

    def get_step_towards(self, current_cell, target_cell):
        row_step =np.sign(target_cell[0] - current_cell[0])
        col_step =np.sign(target_cell[1] - current_cell[1])
        return (current_cell[0] + row_step, current_cell[1] + col_step )


    def get_cells_in_range(self, cell, range_dist):
        cells_in_range = []
        for row in range(cell[0] - range_dist, cell[0] + range_dist + 1):
            for col in range(cell[1] - range_dist, cell[1] + range_dist + 1):
                if 0 <= row < self.dim[0] and 0 <= col < self.dim[1]:
                    cells_in_range.append((row, col))
        return cells_in_range

    def density_remover(self, range_dist: int, max_immune_cells: int):
        for cell in list(self.immuneCells):
            cells_in_range = self.get_cells_in_range(cell, range_dist)
            immune_count = sum(1 for c in cells_in_range if self.immuneLattice[c[0], c[1]] == TKILLER_CELL)
            cancer_count = sum(1 for c in cells_in_range if self.cancerLattice[c[0], c[1]] == CANCER_CELL)

            #if no cancer and more than treshhold in range we remove in steps
            if cancer_count ==0 and immune_count >max_immune_cells:
                excess_count =immune_count -max_immune_cells
                cells_to_remove =self.select_cells_to_remove(cells_in_range, excess_count)

                for remove_cell in cells_to_remove:
                    self._removeImmune(remove_cell)

    def select_cells_to_remove(self, cells_in_range, number_to_remove):
        immune_cells_in_range = [cell for cell in cells_in_range if
                                 self.immuneLattice[cell[0], cell[1]] == TKILLER_CELL]
        random.shuffle(immune_cells_in_range)
        return set(immune_cells_in_range[:number_to_remove])

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

    def get_equilibrium(self):
        """
        Returns the equilibrium of the system, i.e. the relative averge amount of immune cells after the system has reached a steady state. 
        Steady state is defined as when the moving average over 100 timesteps stays within 1% of the previous moving average.

        Returns:
            float: The equilibrium of the system: fraction of immune cells in the system.
        """
        # Calculate the moving average over 100 timesteps
        moving_average = np.convolve(self.immune_cells_over_time, np.ones(100), 'valid') / 100

        # Calculate the difference between the moving average and the previous moving average
        differences = np.diff(moving_average)

        # Calculate the percentage change between the moving average and the previous moving average
        percentage_change = np.abs(differences / moving_average[:-1])

        # Find the index of the first timestep where the percentage change is less than 1%
        equilibrium_index = np.where(percentage_change < 0.01)[0][0]

        return moving_average[equilibrium_index], equilibrium_index
    
    def get_avalanche_sizes(self):
        """
        Returns a list of deviations from the equilibrium of the system after the system has reached a steady state.
        The deviations are measured by the amount of immune cells in the system.
        """
        equilibrium, equilibrium_index = self.get_equilibrium()
        avalanche_sizes = []

        # Calculate deviations from the equilibrium
        for i in range(equilibrium_index, len(self.immune_cells_over_time)):
            avalanche_sizes.append(np.abs(self.immune_cells_over_time[i] - equilibrium))

        return avalanche_sizes
    
    def plot_avalanche_sizes(self):
        """
        Plots the frequency of avalanche sizes of the system after the system has reached a steady state.
        """
        avalanche_sizes = self.get_avalanche_sizes()

        # Fit the power law to the avalanche sizes
        power_avalanche_results = powerlaw.Fit(avalanche_sizes)
        R, p = power_avalanche_results.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Plot the PDF using the powerlaw results on the created axis
        power_avalanche_results.plot_pdf(ax=ax, color='b', linestyle='-', label='Power law fit')
        power_avalanche_results.power_law.plot_pdf(ax=ax, color='r', linestyle='--', label='Fitted power law')

        # Set labels and title using the axis methods
        ax.set_xlabel('Log of Avalanche Size')
        ax.set_ylabel('Log of Frequency')
        ax.set_title('Log-Log Plot of Avalanche Sizes')

        # Add text annotation on the plot
        ax.text(0.3, 0.95, f'Power law vs lognormal={R:.2f} \np-value of fit={p:.2f}', 
            horizontalalignment='center', verticalalignment='top', 
            transform=ax.transAxes, bbox={'facecolor': 'white', 'alpha': 0.5})

        # Add legend and show the plot
        ax.legend()
        plt.savefig('avalanche_sizes.png')


    def timestep(self):
        """
        Propagates the model by 1 step
        """

        for cell in self.cancerCells:
            self.propagateCancerCell(cell)
        for cell in self.immuneCells:
            self.multiTKiller(cell)

        self.cancerCells = self.cancerCells_t1
        self.cancerCells_t1 = set()
        self.immuneCells = self.immuneCells_t1
        self.density_remover(range_dist=5, max_immune_cells=5)
        self.immuneCells_t1 = set()

        self.cancer_emergence()

        immune_cells_at_this_step = self.get_nImmuneCells()
        self.immune_cells_over_time.extend([immune_cells_at_this_step])

        cluster_sizes_at_this_step = self.detect_clusters()
        self.cluster_sizes.extend(cluster_sizes_at_this_step)

        self.time += 1