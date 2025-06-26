# Design CIS 2D color router
# Resolution 25 

# ## 1. Simulation Environment
import copy
import meep as mp
import meep.adjoint as mpa
import numpy as np
import nlopt
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
from scipy.ndimage import label, binary_dilation
import os
from skimage.measure import euler_number
from skimage.measure import label as label_


import wandb
wandb.init(project="colorrouter_GD")
mp.verbosity(1)


design_dir = "./CIS_design_GD/"

# Create directory if it doesn't exist
if not os.path.exists(design_dir):
    os.makedirs(design_dir)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

plt.rcParams["figure.figsize"] = (3.5,3.5)



params = {
    'axes.labelsize':9, # label font size
    'axes.titlesize':9, # title font size
    'xtick.labelsize':9, # x axis tick label font size
    'ytick.labelsize':9, # y axis tick label font size 
    'xtick.direction': 'in', # tick direction (in, out, inout)
    'ytick.direction': 'in', # tick direction (in, out, inout)
    'lines.markersize': 3, # marker size
    'axes.titlepad': 6, # spacing between title and graph
    'axes.labelpad': 4, # spacing between axis label and graph
    'font.size': 9, # font size
    'figure.dpi': 300, # resolution, vector graphics output cleanly regardless of dpi
    'figure.autolayout': True, # automatic layout (all graph elements fit inside figure)
    'xtick.top': True, # show x-axis ticks on top
    'ytick.right': True, # show y-axis ticks on right
    'xtick.major.size': 2, # x-axis tick length
    'ytick.major.size': 2, # y-axis tick length
}



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



def highlight_minimum_island(array, min_label, labeled_array, title):
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
    ax.set_title(title)
    
    # Adjust boundary line thickness and make it more vivid
    boundary_indices = np.argwhere(boundary)
    for y, x in boundary_indices:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='red', facecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    return plt


plt.rcParams.update(params)

# scaling & refractive index
um_scale = 1

Air = mp.Medium(index=1.0)
SiN = mp.Medium(epsilon=4)
SiO2 = mp.Medium(epsilon=2.1)
TiO2 = mp.Medium(epsilon=7)
SiPD = mp.Medium(epsilon=5)

# Design space
design_region_width = 3 # design region width
design_region_height = 3#4 # design region height

# Resolution and size settings
resolution = 21
gapop = 0
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

# Wavelength and frequency settings
wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
frequencies = 1/wavelengths
nf = len(frequencies) # number of frequencies


# Fabrication Constraints settings

minimum_length = 0.895  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
filter_radius = minimum_length
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
Nx = 64
Ny = 64
print('Nx: ', Nx)
print('Ny: ', Ny)

wandb.config.update({"minimum length": minimum_length, "resolution": resolution,
                     "Sx": Sx, "Sy": Sy, "Nx": Nx, "Ny": Ny,
                     "eta_i": eta_i,
                       })
# Design region and material based design region setup
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
    
    try:
        filtered_field = mpa.conic_filter(
            x,
            filter_radius,
            design_region_width,
            design_region_height,
            design_region_resolution,
        )
        x = x.flatten()
        print('filtered_field type: ', type(filtered_field))
        print("filtered_field shape:", getattr(filtered_field, 'shape', 'No shape attribute'))

        if not isinstance(filtered_field, np.ndarray):
                raise TypeError("Expected filtered_field to be an ndarray, got {}".format(type(filtered_field)))

        beta = float(beta)
        eta = float(eta)

        projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    except Exception as e:
        print("Error in mapping function:", e)
        raise
    # projection
    # limit output to 0 ~ 1
    

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
    default_material=Air, # empty space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0) # bloch boundary
)

###############################################################################################################################
# ## 2. Optimization Environment


# Monitor position and size settings (focal point)
monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0)



# Get Fourier transform for region of monitor_size at monitor_position using FourierFields
FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

FourierFields_2= mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)


ob_list = [FourierFields_0, FourierFields_1, FourierFields_2,]
        



fred = []
fgreen = []
fblue = []
# J : Objective function
# Take the squared absolute value of the Ez component at the center of the monitor measured by FourierFields
# [frequency index, monitor index]


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
    decay_by=1e-1, # ratio of field values remaining at monitor
)


evaluation_history = []
cur_iter = [0]
numevl = 1

n = Nx * Ny  # number of parameters




# lower and upper bounds (upper: 1, lower: 0)
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))
LR = 0.01


cur_beta = 2
beta_scale = 2
num_betas = 140
wandb.config.update({
                    'beta_scale': beta_scale,
                    'num_betas': num_betas})


x = np.ones(n)*0.5

wandb.run.name = "min"+str(minimum_length)+"_"+"eta"+str(eta_i)+"_"+"num_betas"+str(num_betas)


for iters in range(num_betas):
    x [x>1] = 1
    x [x<0] = 0

    f0, dJ_du = opt([mapping(x, eta_i, cur_beta)])
    adjoint_gradient = np.sum(dJ_du, axis =1 )
    reshaped_gradients = adjoint_gradient.reshape(adjoint_gradient.shape[0], -1)

    # Computing the norm of each reshaped row
    norms = np.linalg.norm(reshaped_gradients, axis=1)

    # Calculating the mean of the norms
    adjgrad_norm = norms.mean()

    
    print(adjoint_gradient)
    print(type(adjoint_gradient))
    learning_rate = LR/adjgrad_norm
    
    
    x_image = x.reshape(Nx, Ny,1)
    print(x_image.shape)

    wandb.log({
                    'fom ': np.real(f0),
                    'adjgrad norm': adjgrad_norm,
                    'learning rate': learning_rate,
                    "generated": [wandb.Image(x_image, caption=f"Iteration {iters}")]

                }),

    x = x + learning_rate * adjoint_gradient
    cur_beta = cur_beta * beta_scale

## final

x [x>0.5] = 1
x [x==0.5] = 1
x [x<0.5] = 0

x_final = mapping(x, eta_i, mp.inf)

f0, dJ_du = opt([x_final])
adjoint_gradient = np.sum(dJ_du, axis =1 )

x_image = x.reshape(Nx, Ny,1)

wandb.log({
                'final_fom ': np.real(f0),
                "generated_final": [wandb.Image(1-x_image, caption=f"generated_final")]

            }),

sensitivity = [0]

sensitivity[0] = adjoint_gradient

fig = plt.figure(figsize=(20,20))
plt.imshow(np.squeeze(np.abs(sensitivity[0].reshape(Nx, Ny))))
plt.xlabel("x")
plt.ylabel("y")
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

wandb.log({
    'fom (final, island deleted1)': fom_island,
    "generated_final, island deleted1": [wandb.Image(1-island_deleted, caption=f"generated_final_island_deleted")],
})
array_converted_processed = delete_islands_with_size_1(1-island_deleted.copy())
array_original = 1- array_converted_processed

min_feature_size2, min_feature_label2, labeled_array2 = find_minimum_feature_size(array_converted_processed)
plt2 = highlight_minimum_island(array_original, min_feature_label2, labeled_array2, title='Minimum Feature 0s')
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
    "highlighted, island deleted2": [wandb.Image(plt2, caption=f"generated_final_island_deleted_highlghted")],

})
plt3 = highlight_minimum_island(array_original, min_feature_label, labeled_array, title='Minimum Feature 1s')


wandb.log({
    'fom (final, island deleted2)': fom_island2,
    "generated_final, island deleted2": [wandb.Image(1-array_original, caption=f"generated_final_island_deleted")],
     "highlighted, final": [wandb.Image(plt3, caption=f"generated_final_island_deleted_highlghted")]
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

