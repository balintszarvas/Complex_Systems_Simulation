import numpy as np
import matplotlib
matplotlib.use('TkAgg') #REMOVE IF NECESSARY, depends on pc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import label, generate_binary_structure

# Model parameters, all this can be tweaked
size =200
initial_cancer_cells =1
initial_t_cells =150  # Original number of T cells
initial_b_cells =150  # Original number of B cells
time_steps =250
proliferation_prob =0.05  # Proliferation probability for cancer cells
destruction_prob =0.5  # Probability of immune cell killing cancer cell
threshold =15  # Threshold for enhanced immune response
immune_response_factor =150  # Multiplicative factor for immune response
immune_cleanup_rate =0.1  # Rate at which immune cells are reduced after the response

# Initialize lattices
cancer_lattice =np.zeros((size, size))
t_cell_lattice =np.zeros((size, size))
b_cell_lattice =np.zeros((size, size))

#random starters
np.random.seed(69)  # For reproducibility6
cancer_indices =np.random.choice(size * size, initial_cancer_cells, replace=False)
t_cell_indices =np.random.choice(size * size, initial_t_cells, replace=False)
b_cell_indices =np.random.choice(size * size, initial_b_cells, replace=False)
cancer_lattice.flat[cancer_indices] =1
t_cell_lattice.flat[t_cell_indices] =1
b_cell_lattice.flat[b_cell_indices] =1

#make the T/Bcells  attack the cancer
def get_direction_towards_cancer(x, y, cancer_lattice, search_range):
    min_dist =float('inf')
    target_dx, target_dy =0, 0

    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            nx,ny =(x +dx) % size,(y + dy) % size
            if cancer_lattice[nx,ny] ==1:
                dist=abs(dx) + abs(dy)
                if dist < min_dist:
                    min_dist =dist
                    target_dx,target_dy =dx,dy

    if min_dist<=search_range and min_dist> 0:
        return np.sign(target_dx),np.sign(target_dy)
    else:
        return np.random.choice([-1,0,1]),np.random.choice([-1,0,1])

#calc for logplot
def cluster_sizes(lattice):
    structure =generate_binary_structure(2, 2)
    labeled, nclusters =label(lattice, structure)
    return np.bincount(labeled.ravel())[1:]

#updates the lattices
def update_lattices(cancer_lattice,t_cell_lattice,b_cell_lattice,flag):
    new_cancer_lattice =np.copy(cancer_lattice)
    new_t_cell_lattice =np.copy(t_cell_lattice)
    new_b_cell_lattice =np.copy(b_cell_lattice)

    if flag:
        temp = 10 #adjust the range in wich the cells attack AFTER enhanced immume response
    else:
        temp =3
    #cancer cells NO WRAP
    for (x, y), value in np.ndenumerate(cancer_lattice):
        if value ==1:
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny =x + dx, y + dy
                if 0 <=nx < size and 0 <=ny < size:
                    if new_cancer_lattice[nx, ny]==0 and np.random.rand() <proliferation_prob:
                        new_cancer_lattice[nx, ny]=1

    #T cell and B cell WRAP
    for (x, y), value in np.ndenumerate(t_cell_lattice):
        if value ==1:
            dx,dy =get_direction_towards_cancer(x,y,cancer_lattice,search_range =temp)
            nx,ny =(x+dx) % size,(y+dy) % size
            if new_cancer_lattice[nx,ny] ==1 and np.random.rand() <destruction_prob:
                new_cancer_lattice[nx,ny] =0
            new_t_cell_lattice[x,y] =0
            new_t_cell_lattice[nx,ny] =1

    for (x,y),value in np.ndenumerate(b_cell_lattice):
        if value ==1:
            dx,dy =get_direction_towards_cancer(x,y,cancer_lattice, search_range=temp)
            nx,ny =(x+dx) % size,(y+dy) % size
            if new_cancer_lattice[nx,ny] ==1 and np.random.rand() < destruction_prob:
                new_cancer_lattice[nx,ny] =0
            new_b_cell_lattice[x,y] =0
            new_b_cell_lattice[nx,ny] =1

        # Enhanced immune response
    if not flag and np.sum(cancer_lattice) >=threshold:
        flag =True
        additional_t_cells =immune_response_factor
        additional_b_cells =immune_response_factor

        # Adding T cells
        new_t_cell_indices =np.random.choice(size*size, additional_t_cells,replace=False)
        for idx in new_t_cell_indices:
            x,y =np.unravel_index(idx,(size, size))
            new_t_cell_lattice[x, y] =1

        # same for B cells
        new_b_cell_indices =np.random.choice(size*size,additional_b_cells,replace=False)
        for idx in new_b_cell_indices:
            x, y =np.unravel_index(idx, (size,size))
            new_b_cell_lattice[x,y] =1

    #ones cancer gone return to old state
    if flag and np.sum(new_cancer_lattice) < 1: #adjust 1 value for different end value
        t_cell_count =np.sum(new_t_cell_lattice)
        b_cell_count =np.sum(new_b_cell_lattice)

        #do T
        if t_cell_count > initial_t_cells:
            t_cell_indices =np.transpose(np.nonzero(new_t_cell_lattice))
            reduction_count =min(int(immune_cleanup_rate * t_cell_count),t_cell_count-initial_t_cells)
            reduction_count =int(reduction_count)
            removal_indices =np.random.choice(len(t_cell_indices),reduction_count,replace=False)
            for i in removal_indices:
                new_t_cell_lattice[t_cell_indices[i][0],t_cell_indices[i][1]] =0
        #do B
        if b_cell_count > initial_b_cells:
            b_cell_indices =np.transpose(np.nonzero(new_b_cell_lattice))
            reduction_count =min(int(immune_cleanup_rate*b_cell_count),b_cell_count - initial_b_cells)
            reduction_count =int(reduction_count)
            removal_indices =np.random.choice(len(b_cell_indices),reduction_count,replace=False)
            for i in removal_indices:
                new_b_cell_lattice[b_cell_indices[i][0],b_cell_indices[i][1]] =0

    return new_cancer_lattice, new_t_cell_lattice, new_b_cell_lattice, flag

# Visualization setup
fig, ax =plt.subplots()
ims =[]
flag =False  # Flag to track immune response triggering
cluster_size_data =[]

# Simulation loop
for t in range(time_steps):
    cancer_lattice, t_cell_lattice, b_cell_lattice, flag =update_lattices(
        cancer_lattice, t_cell_lattice, b_cell_lattice, flag
    )

    # get cluster data
    sizes =cluster_sizes(cancer_lattice)
    cluster_size_data.append(sizes)
    #visualize
    img =np.zeros((size, size, 3))
    img[cancer_lattice ==1] =[1, 0, 0]  # Red for cancer cells
    img[t_cell_lattice ==1] =[0, 1, 0]  # Green for T cells
    img[b_cell_lattice ==1] =[0, 0, 1]  # Blue for B cells
    ims.append([plt.imshow(img, animated=True)])

# animation
ani =animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('cancer_immune_simulation.gif', writer='pillow')
plt.show()

# Plotting cluster size distribution
all_cluster_sizes =np.concatenate(cluster_size_data)
all_cluster_sizes =all_cluster_sizes[all_cluster_sizes > 1]  # Filter out single cells
hist, edges =np.histogram(all_cluster_sizes, bins=np.logspace(np.log10(2), np.log10(all_cluster_sizes.max()), 50))
centers =(edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(10, 6))
plt.loglog(centers, hist, marker='o', linestyle='none')
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster Size')
plt.ylabel('Frequency')
plt.show()
