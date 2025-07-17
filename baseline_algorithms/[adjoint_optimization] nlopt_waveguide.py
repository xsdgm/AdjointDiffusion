import os
import random
import nlopt
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

import meep as mp
import meep.adjoint as mpa
from scipy.ndimage import label, binary_dilation
from skimage.measure import euler_number
from skimage.measure import label as label_
from autograd import numpy as npa
from autograd import tensor_jacobian_product


# Initialize Weights & Biases (WandB)
wandb.init(project="adjoint_waveguide", settings=wandb.Settings(silent=True))

# Create a directory for saving results
directory = 'adjoint/new'
if not os.path.exists(directory):
    os.makedirs(directory)

# Matplotlib configurations
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5, 3.5)
params = {
    'axes.labelsize': 12,        # label font size
    'axes.titlesize': 12,        # title font size
    'xtick.labelsize': 10,       # x-axis tick label font size
    'ytick.labelsize': 10,       # y-axis tick label font size
    'xtick.direction': 'in',     # tick marking direction
    'ytick.direction': 'in',
    'lines.markersize': 3,       
    'axes.titlepad': 6,          
    'axes.labelpad': 4,          
    'font.size': 12,             
    'figure.dpi': 300,           
    'figure.autolayout': True,
    'xtick.top': True,           
    'ytick.right': True,         
    'xtick.major.size': 2,       
    'ytick.major.size': 2        
}
plt.rcParams.update(params)


###############################################################################
#                         Utility / Helper Functions                          #
###############################################################################

def delete_islands_with_size_1(array):
    """
    Function to delete every island (connected component) with size 1 in the array.
    Uses scipy.ndimage.label.
    """
    labeled_array, num_features = label(array)
    islands_to_delete = [i for i in range(1, num_features + 1) 
                         if np.sum(labeled_array == i) == 1]
    for island in islands_to_delete:
        array[labeled_array == island] = 0

    # If 3D array, squeeze the last axis if it's size=1
    if array.ndim > 2:
        array = np.squeeze(array, axis=2)
    return array


def find_minimum_feature_size(array):
    """
    Find the minimum feature size and its label in a binary array.
    Returns (minimum_feature_size, label_of_that_feature, labeled_array).
    """
    labeled_array, num_features = label(array)
    feature_sizes = [(i, np.sum(labeled_array == i)) 
                     for i in range(1, num_features + 1)]
    min_feature = min(feature_sizes, key=lambda x: x[1]) if feature_sizes else (0, 0)
    return min_feature[1], min_feature[0], labeled_array


def highlight_minimum_island(array, min_label, labeled_array):
    """
    Creates a matplotlib plot highlighting the boundary of the island
    with 'min_label' in the given 'array'.
    """
    coords = np.argwhere(labeled_array == min_label)
    if coords.size == 0:
        return array

    boundary_mask = np.zeros_like(array)
    boundary_mask[labeled_array == min_label] = 1
    
    dilated_mask = binary_dilation(boundary_mask)
    boundary = dilated_mask - boundary_mask
    
    fig, ax = plt.subplots()
    ax.imshow(1 - array, cmap='gray')  # black: 1, white: 0
    ax.imshow(boundary, cmap='Reds', alpha=0.5)
    
    # Enhance boundary appearance
    boundary_indices = np.argwhere(boundary)
    for y, x in boundary_indices:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                             edgecolor='red', facecolor='red', linewidth=1)
        ax.add_patch(rect)
    return plt


###############################################################################
#                       Meep and Adjoint Setup                                #
###############################################################################

# Meep setup
mp.verbosity(0)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

resolution = 21
Sx = 10
Sy = 10
cell_size = mp.Vector3(Sx, Sy)
pml_layers = [mp.PML(2.0)]

fcen = 1 / 1.55
width = 0.2
fwidth = width * fcen
source_center = [-2.7, 0, 0]
source_size = mp.Vector3(0, 2, 0)
kpoint = mp.Vector3(1, 0, 0)

src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [
    mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        size=source_size,
        center=source_center,
    )
]

# Design region resolution, grid size, and MaterialGrid
design_region_resolution = 21
Nx = 64
Ny = 64

design_variables = mp.MaterialGrid(
    mp.Vector3(Nx, Ny),
    SiO2,
    Si,
    grid_type="U_MEAN"
)

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0))
)

###############################################################################
#                          Geometry Definition                                #
###############################################################################
geometry = [
    mp.Block(
        center=mp.Vector3(x=-Sx / 4),
        material=Si,
        size=mp.Vector3(Sx / 2, 1, 0)
    ),  # horizontal waveguide

    mp.Block(
        center=mp.Vector3(y=Sy / 4),
        material=Si,
        size=mp.Vector3(1, Sy / 2, 0)
    ),  # vertical waveguide

    mp.Block(
        center=design_region.center,
        size=design_region.size,
        material=design_variables
    ),
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    eps_averaging=True,
    subpixel_tol=1e-4,
    resolution=resolution,
)

###############################################################################
#                         Define Observables                                  #
###############################################################################
TE_front = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
)
TE_top = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)), mode=1
)
TE_bot = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)), mode=1, forward=False
)
TE_0 = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
)

# We consider TE_0 (source side) and TE_top as the metrics for objective
ob_list = [TE_0, TE_top]

def J(source, top):
    """Objective function: maximize |top/source|^2."""
    return npa.abs(top / source) ** 2

###############################################################################
#                           Optimization Setup                                #
###############################################################################
minimum_length = 0.895
eta_i = 0.5
eta_e = 0.75
eta_d = 1 - eta_e
filter_radius = minimum_length  # You can also use mpa.get_conic_radius_from_eta_e(...)
design_region_width = 3
design_region_height = 3

wandb.config.update({
    "minimum length": minimum_length,
    "resolution": resolution,
    "Sx": Sx,
    "Sy": Sy,
    "Nx": Nx,
    "Ny": Ny,
    "eta_i": eta_i,
    "eta_e": eta_e,
    "eta_d": eta_d
})

def mapping(x, eta, beta):
    """
    Mapping function that:
    1) Applies a conic filter (like a blur) to implement a minimum length scale.
    2) Projects the values using tanh_projection.
    3) Returns an array in [0,1].
    """
    x_filtered = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width, 
        design_region_height,
        design_region_resolution
    )
    projected_field = mpa.tanh_projection(x_filtered, beta, eta)
    return projected_field.flatten()

# Create the optimization problem
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
)

# Initialize design variables
x = 0.5 * np.ones((Nx * Ny,))
opt.update_design([x])

evaluation_history = []
sensitivity = [0]

# Set up optimization algorithms
algorithm1 = nlopt.LD_MMA
algorithm2 = nlopt.LD_SLSQP
algorithm_dict = {
    algorithm1: 'MMA',
    algorithm2: 'SLSQP',
}

def algorithm_name(alg):
    return algorithm_dict.get(alg, "Unknown Algorithm")

algorithm = algorithm2

n = Nx * Ny
MAXEVAL = 250
eta = 0.5
k = 0
R = 8

def f(x, grad, beta):
    """
    The callback function for the optimizer.
    Evaluates the objective, computes gradients, logs to WandB,
    and saves snapshots.
    """
    global k

    # Map the design variable with current threshold parameters
    x_mapped = mapping(x, eta, beta)
    
    # Evaluate objective & gradient
    f0, dJ_du = opt([x_mapped])
    adjoint_gradient = dJ_du
    reshaped_gradients = adjoint_gradient.reshape(adjoint_gradient.shape[0], -1)
    
    # Compute norm for logging
    norms = np.linalg.norm(reshaped_gradients, axis=1)
    adjgrad_norm = norms.mean()
    
    print('fom: ', f0)
    print('adjgrad_norm: ', adjgrad_norm)
    
    # Log images to WandB
    wandb.log({
        "fom": f0,
        "adjgrad_norm": adjgrad_norm,
        "generated": [wandb.Image(1 - x_mapped.reshape(Nx, Ny),
                     caption='step' + str(k) + '_fom' + str(f0)[1:5])]
    }, step=k)
    
    # f0 is an array of length 1, so take its first (only) element
    f0_scalar = f0[0]
    
    # If gradient is requested by the optimizer
    if grad.size > 0:
        grad[:] = tensor_jacobian_product(mapping, 0)(x, eta_i, beta, dJ_du)
    
    evaluation_history.append(np.real(f0_scalar))
    print(evaluation_history)
    sensitivity[0] = dJ_du
    
    # Update design in meep-adjoint
    opt.update_design([x_mapped])
    opt.plot2D()
    
    # Save figure and data
    filename_prefix = f'adjoint/new/{str(k).zfill(2)}_{str(f0_scalar)[:5]}'
    plt.savefig(filename_prefix + '.png')
    np.save(filename_prefix + '.npy', x)
    
    k += 1
    return np.real(f0_scalar)

cur_beta = 2
beta_scale = 2
num_betas = 7
update_factor = 20

wandb.run.name = (f"min{minimum_length}_eta_i{eta_i}_"
                  f"{algorithm_name(algorithm)}_num_betas{num_betas}")

for iters in range(num_betas):
    print("current beta: ", cur_beta)
    
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    print('cur_beta: ', cur_beta)
    
    # If beta is large enough, effectively no more projection smoothing
    if cur_beta >= 2 ** (num_betas + 1):
        solver.set_max_objective(lambda a, g: f(a, g, mp.inf))
        solver.set_maxeval(1)
    else:
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor)
    
    np.savetxt('adjoint/new/history.txt', evaluation_history)
    x[x > 1] = 1
    x[x < 0] = 0
    x = solver.optimize(x)
    
    cur_beta *= beta_scale
    wandb.log({'cur_beta': cur_beta}, step=k)
    wandb.config.update({
        'update_factor': update_factor,
        'beta_scale': beta_scale,
        'num_betas': num_betas
    })
    wandb.config.update({'algorithm': algorithm_name(algorithm)})

# Final thresholding of the design : must be deleted!!
#x[x >= 0.5] = 1
#x[x < 0.5] = 0

x_final = mapping(x, eta_i, mp.inf)
f0, dJ_du = opt([x_final])
sensitivity[0] = dJ_du

wandb.log({
    'final_fom': np.real(f0),
    "generated_final": [wandb.Image(1 - design_variables.weights.reshape(Nx, Ny))],
})

###############################################################################
#                      Plot Optimization Progress                             #
###############################################################################
plt.figure()
plt.plot(np.array(evaluation_history), "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("efficiency")
plt.savefig('adjoint/new/log.png')

fig = plt.figure(figsize=(5, 5))
plt.imshow(np.squeeze(np.abs(sensitivity[0].reshape(Nx, Ny))))
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("adjoint/new/sensitivity.png")

cmap = plt.cm.get_cmap()
colormapping = plt.cm.ScalarMappable(cmap=cmap)
cbar = fig.colorbar(colormapping, ax=plt.gca())
wandb.log({"sensitivity": plt})

###############################################################################
#                       Post-Processing of the Design                         #
###############################################################################
x_final = x_final.reshape(Nx, Ny)

# 1) Delete small islands (size=1)
island_deleted = delete_islands_with_size_1(x_final.copy())
min_feature_size, min_feature_label, labeled_array = find_minimum_feature_size(island_deleted)

fom_island, _ = opt([island_deleted.flatten()])
print('min feature size:', min_feature_size)
print('min feature label:', min_feature_label)
print('labeled array shape:', labeled_array.shape)
print('island_deleted shape:', island_deleted.shape)

wandb.log({
    'fom (final, island deleted1)': fom_island,
    "generated_final, island deleted1": [wandb.Image(1 - island_deleted)],
})

# 2) Convert array to further process
array_converted_processed = delete_islands_with_size_1(1 - island_deleted.copy())
array_original = 1 - array_converted_processed

min_feature_size2, min_feature_label2, labeled_array2 = find_minimum_feature_size(array_converted_processed)
plt2 = highlight_minimum_island(array_original, min_feature_label2, labeled_array2)

euler_number_original = euler_number(array_original, connectivity=1)
_, num_islands1 = label_(island_deleted, connectivity=1, return_num=True)
num_holes1 = num_islands1 - euler_number_original

euler_number_converted = euler_number(array_converted_processed, connectivity=1)
_, num_islands2 = label_(array_converted_processed, connectivity=1, return_num=True)
num_holes2 = num_islands2 - euler_number_converted

fom_island2, _ = opt([array_original.flatten()])

wandb.log({"highlighted, island deleted2": [wandb.Image(plt2)]})

plt3 = highlight_minimum_island(array_original, min_feature_label, labeled_array)

wandb.log({
    'fom (final, island deleted2)': fom_island2,
    "generated_final, island deleted2": [wandb.Image(1 - array_original)],
    "highlighted, final": [wandb.Image(plt3)]
})

print("array 0,0: ", array_original[0, 0])

wandb.log({
    'mfs (original)': min_feature_size,
    'mfs (converted)': min_feature_size2,
    'number of islands (original)': num_islands1,
    'number of islands (converted)': num_islands2,
    'number of holes (original)': num_holes1,
    'number of holes (converted)': num_holes2,
    'euler_number with 1 connectivity (original)': euler_number_original,
    'euler_number with 1 connectivity (converted)': euler_number_converted,
})
