import meep as mp
import meep.adjoint as mpa

from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import matplotlib.pyplot as plt
import nlopt
import torch
import numpy as np

from scipy.ndimage import label, binary_dilation
import os
from skimage.measure import euler_number
from skimage.measure import label as label_

import wandb
wandb.init(project = "adjoint_waveguide_GD", settings=wandb.Settings(silent=True))

directory = 'adjoint/new'

import os
if not os.path.exists(directory):
    os.makedirs(directory)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

plt.rcParams["figure.figsize"] = (3.5,3.5)



params = {
    'axes.labelsize':9, # label 폰트 크기
    'axes.titlesize':9, # 타이틀 폰트 크기
    'xtick.labelsize':9, # x 축 tick label 폰트 크기
    'ytick.labelsize':9, # y 축 tick label 폰트 크기 
    'xtick.direction': 'in', # 눈금 표시 방향 (in, out, inout)
    'ytick.direction': 'in', # 눈금 표시 방향 (in, out, inout)
    'lines.markersize': 3, # 마커 사이즈
    'axes.titlepad': 6, # 타이틀과 그래프 사이의 간격
    'axes.labelpad': 4, # 축 label과 그래프 사이의 간격
    'font.size': 9, # font 크기
    #'font.sans-serif': 'Arial', # font 설정
    'figure.dpi': 300, # 해상도, vector그래픽의 경우 dpi에 상관없이 깔끔하게 출력됨
    'figure.autolayout': True, # 레이아웃 자동 설정 (그래프의 모든 요소가 figure 내부에 들어가도록 설정)
    'xtick.top': True, # 그래프 위쪽 x축 눈금 표시
    'ytick.right': True, # 그래프 오른쪽 y축 눈금 표시
    'xtick.major.size': 2, # x축 눈금의 길이
    'ytick.major.size': 2, # y축 눈금의 길이
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
    ax.imshow(1-array, cmap='gray')   # black: 1, while: 0
    ax.imshow(boundary, cmap='Reds', alpha=0.5)  # Overlay the red boundary with higher alpha
    ax.set_title(title)
    
    # Adjust boundary line thickness and make it more vivid
    boundary_indices = np.argwhere(boundary)
    for y, x in boundary_indices:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='red', facecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    return plt

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

design_region_resolution = 21
Nx = 64#design_region_resolution +1
Ny = 64#design_region_resolution +1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables, volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0))
)

"""
        mp.Block(
            center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
        ),  # vertical waveguide
"""
geometry = [
    mp.Block(
        center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
    ),  # horizontal waveguide
    #mp.Block(
    #        center=mp.Vector3(x=Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
    #),  # vertical waveguide
    mp.Block(
            center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
        ),  # vertical waveguide
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),  # design region
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables,
    #         e1=mp.Vector3(x=-1).rotate(mp.Vector3(z=1), np.pi/2), e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), np.pi/2))
    #
    # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; We give an alternative approach to impose symmetry in later tutorials.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
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

TE_front = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
)
TE_top = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)), mode=1
)
TE_bot = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)), mode=1, forward = False
)
TE_0 = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
)
#ob_list = [TE0, TE_top]
ob_list = [TE_0, TE_top]

def J (source, top):
    return npa.abs(top / source) ** 2

'''
def conic_filter(x, R, Nx, Ny):
    x_new = np.zeros_like(x)
    
    for i in range(Nx):
        for j in range(Ny):
            weights_sum = 0
            values_sum = 0
            for m in range(Nx):
                for n in range(Ny):
                    d = np.sqrt((i - m)**2 + (j - n)**2)
                    if d <= R:
                        weight = max(0, 1 - d/R)
                        weights_sum += weight
                        values_sum += weight * x[m * Ny + n]
            
            x_new[i * Ny + j] = values_sum / weights_sum
            
    return x_new
'''


minimum_length = 0.895  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.65  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = minimum_length#mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_width = 3
design_region_height = 3

wandb.config.update({"minimum length": minimum_length, "resolution": resolution,
                     "Sx": Sx, "Sy": Sy, "Nx": Nx, "Ny": Ny,
                     "eta_i": eta_i, "eta_e": eta_e, "eta_d": eta_d
                       })

def mapping (x, eta, beta):
    # filter 씌우기 (블러처리느낌)

    x = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width, 
        design_region_height,
        design_region_resolution
    )
    x = x.flatten ()
    # 출력값 .1~ 1으로 제한

    print('filtered_field type: ', type(x))
    print("filtered_field shape:", getattr(x, 'shape', 'No shape attribute'))

    if not isinstance(x, np.ndarray):
            raise TypeError("Expected filtered_field to be an ndarray, got {}".format(type(filtered_field)))

    beta = float(beta)
    eta = float(eta)
    projected_field = mpa.tanh_projection(x, beta, eta)
    # interpolate to actual materials
    return projected_field.flatten ()

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
)

#struct_random = np.random.choice([0, 1], size=(Nx*Ny))

import random
x = 0.5 * np.ones((Nx * Ny,))
#x0 = np.random.randint(2, size= Nx*Ny)
#x0 = np.random.rand((Nx*Ny))
opt.update_design([x])

evaluation_history = []
sensitivity = [0]




n = Nx*Ny
LR = 1
cur_beta = 2
beta_scale = 2
num_betas = 140


wandb.run.name = "min"+str(minimum_length)+"_"+"eta"+str(eta_i)+"_"+"num_betas"+str(num_betas)+"_beta_scale"+str(beta_scale)+"_LR"+str(LR)


wandb.config.update({
                    'beta_scale': beta_scale,
                    'num_betas': num_betas})

x = np.ones(n)*0.5
for iters in range(num_betas):
    x[x>1] = 1
    x[x<0] = 0
    #print('cur beta: ', cur_beta)
    print('x: ', x)
    f0, dJ_du = opt([mapping(x, eta_i, cur_beta)])
    adjoint_gradient = dJ_du
    reshaped_gradients = adjoint_gradient.reshape(adjoint_gradient.shape[0], -1)

    # Computing the norm of each reshaped row
    norms = np.linalg.norm(reshaped_gradients, axis=1)

    # Calculating the mean of the norms
    adjgrad_norm = norms.mean()
    print(adjoint_gradient)
    print(type(adjoint_gradient))
    
    
    x_image = x.reshape(Nx, Ny,1)
    print(x_image.shape)

    learning_rate = LR/ adjgrad_norm
    #learning_rate = LR
    wandb.log({
                    'fom ': f0,
                    'adjgrad norm': adjgrad_norm,
                    'learning rate': learning_rate,
                    "generated": [wandb.Image(x_image, caption=f"Iteration {iters}")]

                }),
    x = x + learning_rate * adjoint_gradient
    # what if learning_rate = learning_rate / 
    cur_beta = cur_beta * beta_scale



## final

x [x>0.5] = 1
x [x==0.5] = 1
x [x<0.5] = 0

x_final = mapping(x, eta_i, mp.inf)

f0, dJ_du = opt([x_final])
sensitivity[0] = dJ_du
x_image = x.reshape(Nx, Ny,1)

wandb.log({
                'final_fom ': np.real(f0),
                "generated_final": [wandb.Image(x_image, caption=f"generated_final")]

            }),

fig = plt.figure(figsize=(20,20))
plt.imshow((np.squeeze(np.abs(sensitivity[0].reshape(Nx, Ny)))))#(np.rot90(np.squeeze(np.abs(sensitivity[0].reshape(Nx, Ny)))));
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("adjoint/new/sensitivity.png")
cmap = plt.cm.get_cmap()
colormapping = plt.cm.ScalarMappable(cmap=cmap)
cbar = fig.colorbar(colormapping, ax=plt.gca())
wandb.log({"sensitivity": plt})



x_final = x

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
    "generated_final, island deleted1": [wandb.Image(1-island_deleted, caption=f"generated_final_island_deleted")],
   # "highlighted, island deleted1": [wandb.Image(plt1, caption=f"generated_final_island_deleted_highlghted")]
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


print("DONE")

