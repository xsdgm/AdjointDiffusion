# === CIS 2D Color Router Design — Resolution 25 ===

# === 1. Imports & Setup ===
import os
import copy
import numpy as np
import nlopt
import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import torch as th
from scipy.ndimage import label, binary_dilation
from skimage.measure import euler_number, label as sklabel
import wandb
import matplotlib.pyplot as plt

# Initialize Weights & Biases
wandb.init(project="colorrouter_adjoint")

# Remove previous log file
if os.path.exists('lists.pkl'):
    os.remove('lists.pkl')

# Matplotlib defaults
plt.rcParams.update({
    'figure.figsize': (3.5, 3.5),
    'figure.dpi': 300,
    'figure.autolayout': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.titlepad': 6,
    'axes.labelpad': 4,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.markersize': 3,
})

# === 2. Utility Functions ===

# Function to delete every island with size 1
def delete_islands_with_size_1(array):
    # Label the connected components
    labeled_array, num_features = label(array)
    
    # Identify the features (islands) with size 1
    islands_to_delete = [i for i in range(1, num_features + 1) if np.sum(labeled_array == i) == 1]
    
    # Delete these islands
    for island in islands_to_delete:
        array[labeled_array == island] = 0
    
    if array.ndim > 2:
        array = np.squeeze(array, axis=2)
    else:
        pass
    return array


# Function to find the minimum feature size and its location
def find_minimum_feature_size(array):
    # Label the connected components
    labeled_array, num_features = label(array)
    
    # Get the sizes of all features
    feature_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    
    # Find the minimum feature size and its label
    min_feature = min(feature_sizes, key=lambda x: x[1]) if feature_sizes else (0, 0)
    
    return min_feature[1], min_feature[0], labeled_array



def highlight_minimum_island(array, min_label, labeled_array):
    # Find the coordinates of the minimum island
    coords = np.argwhere(labeled_array == min_label)
    if coords.size == 0:
        return array
    
    # Create an empty mask for the boundary
    boundary_mask = np.zeros_like(array)
    
    # Fill the mask with the minimum island
    boundary_mask[labeled_array == min_label] = 1
    
    # Dilate the mask and subtract the original to get the boundary
    dilated_mask = binary_dilation(boundary_mask)
    boundary = dilated_mask - boundary_mask
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(1-array, cmap='gray')   # black: 1, white: 0
    ax.imshow(boundary, cmap='Reds', alpha=0.5)  # Overlay the red boundary with higher alpha
    
    # Adjust boundary line thickness and make it more vivid
    boundary_indices = np.argwhere(boundary)
    for y, x in boundary_indices:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='red', facecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    return plt

# === 3. Algorithm Selection ===
ALGO_MMA   = nlopt.LD_MMA
ALGO_SLSQP = nlopt.LD_SLSQP
ALGORITHMS = {
    ALGO_MMA:   'MMA',
    ALGO_SLSQP: 'SLSQP',
}

def get_algorithm_name(algorithm) -> str:
    """
    Return the human-readable name of an NLopt algorithm.
    """
    return ALGORITHMS.get(algorithm, 'Unknown')

# Select optimization algorithm
algorithm = ALGO_MMA



mp.verbosity(1)


design_dir = "./CIS_design/"

# Create directory if it doesn't exist
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

# scaling & refractive index
um_scale = 1

mp.verbosity(0)

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
TiO2 = mp.Medium(epsilon=7)
SiPD = mp.Medium(epsilon=5)

# Design space
design_region_width = 3 # Design region width
design_region_height = 3#4 # Design region height

# Resolution and size settings
resolution = 21
gapop = 0 ####################################################################################################
air_gap = 0
dti = 0.4
subpixelsize = design_region_width/3 - dti
if gapop == 1:
    air_gap = dti/2
PDsize = 2
Lpml = 0.5 # PML region size
pml_layers = [mp.PML(thickness = Lpml, direction = mp.Y)]
Sourcespace = 2



# Total space
Sx = design_region_width
Sy = PDsize + design_region_height + Sourcespace + Lpml
cell_size = mp.Vector3(Sx, Sy)

# Wavelength, frequency settings
wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# Fabrication Constraints settings

minimum_length = 0.224  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.75  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = minimum_length#mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)
print(filter_radius)


# Source settings

width = 0.4

fcen_red = 1/(0.65*um_scale)
fwidth_red = fcen_red * width

fcen_green = 1/(0.55*um_scale)
fwidth_green = fcen_green * width

fcen_blue = 1/(0.45*um_scale)
fwidth_blue = fcen_blue * width

src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)

src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=fwidth_green, is_integrated=True)

src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=fwidth_blue, is_integrated=True)

source_center = mp.Vector3(0, Sy/ 2 - Lpml - Sourcespace / 2, 0) # Source position
source_size = mp.Vector3(Sx, 0, 0)

source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center,),]




# Design region pixels - determined by resolution and design region
Nx = 64#int(round(design_region_resolution * design_region_width)) + 1
Ny = 64#int(round(design_region_resolution * design_region_height)) + 1
print('Nx: ', Nx)
print('Ny: ', Ny)


wandb.config.update({"minimum length": minimum_length, "resolution": resolution,
                     "Sx": Sx, "Sy": Sy, "Nx": Nx, "Ny": Ny,
                     "eta_i": eta_i, "eta_e": eta_e, "eta_d": eta_d
                       })
# Design region and material-based design region setup
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, - Sy /2 + PDsize + design_region_height / 2, 0),
        size=mp.Vector3(design_region_width-air_gap*2, design_region_height, 0),
    ),
)



# Use conic_filter function and simple_2d_filter function from filter.py
def mapping(x, eta, beta):
    # filter
    x = x.flatten()
    
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        design_region_resolution,
    )
    print('filtered_field: ', type(filtered_field))

    # projection
    # Limit output values to 0 ~ 1
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return projected_field.flatten()


# Create Block with same size as design region
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(Sx, PDsize, 0), material=SiO2
    ),
    # Use when DTI is present
    mp.Block(
        center=mp.Vector3(-design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    ),
    mp.Block(
        center=mp.Vector3(design_region_width/3, -Sy/2 + PDsize/2, 0), size=mp.Vector3(subpixelsize, PDsize, 0), material=SiPD
    )
]



# Meep simulation setup
sim = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # Empty space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0) # bloch boundary
)

###############################################################################################################################
# ## 2. Optimization Environment


#Monitor position and size settings (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0)



# Get Fourier transform for the area of monitor_size at monitor_position through FourierFields
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2= mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)


ob_list = [FourierFields_0, FourierFields_1, FourierFields_2,]
        



fred = []
fgreen = []
fblue = []
# J : Objective function
# Take the squared absolute value of the Ez component at the center of the monitor measured by FourierFields
# [frequency index, moniter index]


flag = 0
def J_0(fields_0, fields_1, fields_2):
    red = npa.sum(npa.abs(fields_0[21:30,:]) **2)
    green = npa.sum(npa.abs(fields_1[11:20,:]) ** 2) 
    blue = npa.sum(npa.abs(fields_2[1:10,:]) ** 2) 

    
    
    redfactor = 1
    greenfactor = 1
    bluefactor = 1

    if isinstance(red, np.floating):
        red_ = red
    else:
        red_ = red._value

    if isinstance(green, np.floating):
        green_ = green
    else:
        green_ = green._value

    if isinstance(blue, np.floating):
        blue_ = blue
    else:
        blue_ = blue._value


    
    fred.append(red_/redfactor)
    fgreen.append(green_/greenfactor)
    fblue.append(blue_/bluefactor)

    return blue/bluefactor + green/greenfactor + red/redfactor


# Optimization settings
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J_0],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-3, # Ratio of field values remaining in monitor
)


evaluation_history = []
cur_iter = [0]
numevl = 0

def f(v, gradient, beta):
    global numevl
    print("Current iteration: {}".format(cur_iter[0] + 1))
    print("x: ", type(v))
    x_image = x.reshape(Nx, Ny,1)
    print(x_image.shape)

    f0, dJ_du = opt([mapping(v, eta_i, beta)])  # compute objective and gradient
    print("FoM: ", f0)
    # f0, dJ_du = opt()
    adjoint_gradient = np.sum(dJ_du, axis =1 )
    reshaped_gradients = adjoint_gradient.reshape(adjoint_gradient.shape[0], -1)

    # Computing the norm of each reshaped row
    norms = np.linalg.norm(reshaped_gradients, axis=1)

    # Calculating the mean of the norms
    adjgrad_norm = norms.mean()

    wandb.log({
                    "generated": [wandb.Image(1-x_image, caption=f"Iteration {numevl}")],
                    'fom ': f0,
                    'adjgrad norm': adjgrad_norm
                }, step = numevl),
    
    # Adjoint gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, np.sum(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))

    np.savetxt(design_dir+"structure_0"+str(numevl) +".txt", design_variables.weights)

    numevl += 1

    cur_iter[0] = cur_iter[0] + 1
    
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    

    return np.real(f0)

###############################################################################################################################

# ## 3. Algorithm select



n = Nx * Ny  # number of parameters

# Initial guess - random initial starting value
#x = np.random.uniform(0.3, 0.7, n)

x = np.ones(n)*0.5

# lower and upper bounds (upper bound: 1, lower bound: 0)
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# Optimization parameter
cur_beta = 2
beta_scale = 2
num_betas = 5
update_factor = 20  # number of iterations between beta updates
ftol = 1e-5


wandb.run.name = "min"+str(minimum_length)+"_"+"eta_i"+str(eta_i)+"_"+str(algorithm_name(algorithm))+"_"+"num_betas"+str(num_betas)


for iters in range(num_betas):
    print("current beta: ", cur_beta)
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    if cur_beta >=2**(num_betas+1):   # 2^5
        solver.set_max_objective(lambda a, g: f(a, g, mp.inf))
        solver.set_maxeval(1) # Set the maximum number of function evaluations
    else:
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    #solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    print(x)
    x_ = copy.copy(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration
    wandb.log({
                    'cur_beta': cur_beta
    }, step = numevl)
    wandb.config.update(   {'update_factor': update_factor,
                    'beta_scale': beta_scale,
                    'num_betas': num_betas})

    wandb.config.update({'algorithm': algorithm_name(algorithm)})

###############################################################################################################################


# ## 4. Save result

#np.save("evaluation", evaluation_history)
np.savetxt(design_dir+"evaluation.txt", evaluation_history)

# FoM plot

plt.figure()

plt.plot(evaluation_history, "k-")
plt.grid(False)
plt.tick_params(axis='x', direction='in', pad = 5)
plt.tick_params(axis='y', direction='in', pad = 10)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"FoMresult.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


def extract_elements(lst):
    # Create a list to store the results.
    result = []

    # Iterate through the length of the list, extracting elements at indices that are multiples of 5.
    for i in range(0, len(lst), 5):
        result.append(lst[i])

    return result

# RGB FoM plot

fred = extract_elements(fred)
fgreen = extract_elements(fgreen)
fblue = extract_elements(fblue)


columns = ['red', 'green', 'blue']
wandb.log({"Intensity": wandb.plot.line_series(
        xs = range(len(fred)),
        ys = [fred, fgreen, fblue],
        keys = columns,
        xname = "Step"
    )})

plt.figure()

plt.plot(fred, "r-")
plt.plot(fgreen, "g-")
plt.plot(fblue, "b-")
plt.grid(False)
plt.tick_params(axis='x', direction='in', pad = 5)
plt.tick_params(axis='y', direction='in', pad = 10)
plt.xlabel("Iteration")
plt.ylabel("FoM")
plt.savefig(design_dir+"FoMresultr.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure





# Last design plot

plt.imshow(design_variables.weights.reshape(Nx, Ny), cmap='binary')
plt.colorbar()

x_final = mapping(x, eta_i, mp.inf)

f0, dJ_du = opt([x_final])



np.savetxt(design_dir+"lastdesign.txt", design_variables.weights)
np.save(design_dir+"lastdesign.npy", design_variables.weights)
wandb.log({
            "generated_final": [wandb.Image(1-design_variables.weights.reshape(Nx, Ny))],
            'final_fom ': np.real(f0),
        })

plt.savefig(design_dir+"lastdesign.png")
plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure


sensitivity= np.sum(dJ_du, axis =1 )

fig = plt.figure(figsize=(20,20))
plt.imshow(np.squeeze(np.abs(sensitivity.reshape(Nx, Ny))))#(np.rot90(np.squeeze(np.abs(sensitivity[0].reshape(Nx, Ny)))));
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("sensitivity.png")
cmap = plt.cm.get_cmap()
colormapping = plt.cm.ScalarMappable(cmap=cmap)
cbar = fig.colorbar(colormapping, ax=plt.gca())
wandb.log({"sensitivity": plt})




x_final = x_final.reshape(Nx, Ny)

island_deleted = delete_islands_with_size_1(x_final.copy())
min_feature_size, min_feature_label, labeled_array = find_minimum_feature_size(island_deleted)

fom_island, _ = opt([
    island_deleted.flatten()])
print('min feature size: ',  min_feature_size)
print('min feature label: ', min_feature_label)
print('labeled array: ', labeled_array)
print('island_deleted: ', island_deleted.shape)

#plt1 = highlight_minimum_island(island_deleted, min_feature_label, labeled_array, title='Minimum Feature 1s')

wandb.log({
    'fom (final, island deleted1)': fom_island,
    "generated_final, island deleted1": [wandb.Image(1-island_deleted)],
   # "highlighted, island deleted1": [wandb.Image(plt1, caption=f"generated_final_island_deleted_highlghted")]
})
array_converted_processed = delete_islands_with_size_1(1-island_deleted.copy())
array_original = 1- array_converted_processed

min_feature_size2, min_feature_label2, labeled_array2 = find_minimum_feature_size(array_converted_processed)
plt2 = highlight_minimum_island(array_original, min_feature_label2, labeled_array2)
euler_number_original = euler_number(array_original, connectivity=1)
_, num_islands1 = label_(island_deleted, connectivity=1, return_num=True)
num_holes1 = num_islands1 - euler_number_original

euler_number_converted = euler_number(array_converted_processed, connectivity=1)
_, num_islands2 = label_(array_converted_processed, connectivity=1, return_num=True)
num_holes2 = num_islands2 - euler_number_converted


fom_island2, _ = opt([
    array_original.flatten()]
)
wandb.log({
    "highlighted, island deleted2": [wandb.Image(plt2)],

})
plt3 = highlight_minimum_island(array_original, min_feature_label, labeled_array)


wandb.log({
    'fom (final, island deleted2)': fom_island2,
    "generated_final, island deleted2": [wandb.Image(1-array_original)],
     "highlighted, final": [wandb.Image(plt3)]
})

print("array 0,0: ", array_original[0,0])


wandb.log({
                'mfs (original)': min_feature_size,
                'mfs (converted)': min_feature_size2,
                'number of islands (original)': num_islands1,
                'number of islands (converted)':num_islands2,
                'number of holes (original)' : num_holes1,
                'number of holes (converted)': num_holes2,
                'euler_number with 1 connectivity (original)': euler_number_original,
                'euler_number with 1 connectivity (converted)': euler_number_converted,
            }) 

fig2 = plt.figure(figsize=(20, 8))

intensities_R = np.abs(opt.get_objective_arguments()[0][:,0]) ** 2
plt.subplot(1,3,1)
plt.plot(wavelengths/um_scale, intensities_R, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()


intensities_G = np.abs(opt.get_objective_arguments()[1][:,1]) ** 2
fig3 = plt.figure(figsize=(20, 8))
plt.subplot(1,3,2)
plt.plot(wavelengths/um_scale, intensities_G, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()


intensities_B = np.abs(opt.get_objective_arguments()[2][:,1]) ** 2
plt.subplot(1,3,3)
plt.plot(wavelengths/um_scale, intensities_B, "-o")
plt.grid(True)
# plt.xlim(0.38, 0.78)
plt.xlabel("Wavelength (μm)")
plt.ylabel("|Ez|^2 intensity (a.u.)")
# plt.show()
plt.savefig(design_dir+"FinalEz.png")
wandb.log({"FinalEz": plt})

plt.cla()   # clear the current axes
plt.clf()   # clear the current figure
plt.close() # closes the current figure
