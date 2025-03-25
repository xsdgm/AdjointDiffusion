import matplotlib.pyplot as plt
import pickle
import os

# Configure global matplotlib settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5, 3.5)

params = {
    'axes.labelsize': 12,       # font size for axis labels
    'axes.titlesize': 12,       # font size for plot titles
    'xtick.labelsize': 10,      # font size for x-axis tick labels
    'ytick.labelsize': 10,      # font size for y-axis tick labels
    'xtick.direction': 'in',    # tick direction (in, out, inout)
    'ytick.direction': 'in',    # tick direction (in, out, inout)
    'lines.markersize': 3,      # marker size
    'axes.titlepad': 6,         # padding between title and plot
    'axes.labelpad': 4,         # padding between axis label and plot
    'font.size': 12,            # global font size
    # 'font.sans-serif': 'Arial',  # Uncomment to specify a particular sans-serif font
    'figure.dpi': 300,          # DPI setting (for raster graphics)
    'figure.autolayout': True,  # automatically adjust layout to fit all elements
    'xtick.top': True,          # show top x-axis ticks
    'ytick.right': True,        # show right y-axis ticks
    'xtick.major.size': 2,      # x-axis major tick length
    'ytick.major.size': 2,      # y-axis major tick length
}

# Function to save lists to a pickle file
def save_lists(new_red_list, new_blue_list, new_green_list):
    """
    Appends new red, blue, and green data to existing lists 
    and saves them in 'lists.pkl' as a pickle file.
    """
    if os.path.exists('lists.pkl'):
        with open('lists.pkl', 'rb') as f:
            red_list, blue_list, green_list = pickle.load(f)
    else:
        red_list, blue_list, green_list = [], [], []

    red_list += new_red_list
    blue_list += new_blue_list
    green_list += new_green_list

    with open('lists.pkl', 'wb') as f:
        pickle.dump((red_list, blue_list, green_list), f)

def CIS_sim(struct_np, t, exp_name, prop_dir='',
            save_inter=False, interval=10, flag_last=False):
    """
    CIS Simulation function:
    
    - Takes an array 'struct_np' which is used as the design region's binary (0/1) pattern.
    - Sets up a MEEP simulation for a 3x3 design region on top of an SiO2 substrate block 
      with side blocks of SiPD, aiming to capture transmissions of red, green, and blue 
      frequencies.
    - Saves the summed intensities of the fields in separate lists (red, green, blue),
      which are stored using 'save_lists()'.
    - Returns the figure of merit (sum of the intensities) and its gradient w.r.t. 
      the design variables.
    """

    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import matplotlib.pyplot as plt
    import wandb  # If not using wandb, remove or comment out.
    
    # Ensure structure is clipped between 0 and 1
    struct_np[struct_np > 1] = 1
    struct_np[struct_np < 0] = 0

    red_list = []
    blue_list = []
    green_list = []

    mp.verbosity(0)

    # Define materials
    Air = mp.Medium(index=1.0)
    SiN = mp.Medium(epsilon=4)
    SiO2 = mp.Medium(epsilon=2.1)
    SiPD = mp.Medium(epsilon=5)

    # Basic simulation parameters
    um_scale = 1
    resolution = 21

    # Design region dimensions
    design_region_width = 3
    design_region_height = 3
    
    # Additional geometry
    gapop = 0
    air_gap = 0
    dti = 0.4
    subpixelsize = design_region_width / 3 - dti
    if gapop == 1:
        air_gap = dti / 2
    PDsize = 2
    Lpml = 0.5
    Sourcespace = 2

    # Cell size
    Sx = design_region_width
    Sy = PDsize + design_region_height + Sourcespace + Lpml
    cell_size = mp.Vector3(Sx, Sy)

    # PML settings
    pml_layers = [mp.PML(thickness=Lpml, direction=mp.Y)]

    # Frequency ranges
    wavelengths = np.linspace(0.40 * um_scale, 0.70 * um_scale, 31)
    frequencies = 1 / wavelengths
    nf = len(frequencies)

    # Source frequency setup (red, green, blue)
    width = 0.4
    fcen_red = 1 / (0.65 * um_scale)
    fwidth_red = fcen_red * width

    fcen_green = 1 / (0.55 * um_scale)
    fwidth_green = fcen_green * width

    fcen_blue = 1 / (0.45 * um_scale)
    fwidth_blue = fcen_blue * width

    src_0 = mp.GaussianSource(frequency=fcen_red, fwidth=fwidth_red, is_integrated=True)
    src_1 = mp.GaussianSource(frequency=fcen_green, fwidth=fwidth_green, is_integrated=True)
    src_2 = mp.GaussianSource(frequency=fcen_blue, fwidth=fwidth_blue, is_integrated=True)

    source_center = mp.Vector3(0, Sy / 2 - Lpml - Sourcespace / 2, 0)
    source_size = mp.Vector3(Sx, 0, 0)
    source = [
        mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center),
        mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center),
        mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center),
    ]

    # Grid dimensions
    Nx = 64
    Ny = 64

    # Create a MaterialGrid for the design region
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), 
        SiO2, 
        SiN, 
        grid_type="U_MEAN"
    )

    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, -Sy/2 + PDsize + design_region_height / 2, 0),
            size=mp.Vector3(design_region_width, design_region_height, 0),
        ),
    )

    # Geometry
    geometry = [
        mp.Block(
            center=design_region.center, 
            size=design_region.size, 
            material=design_variables
        ),
        mp.Block(
            center=mp.Vector3(0, -Sy/2 + PDsize/2, 0),
            size=mp.Vector3(Sx, PDsize, 0),
            material=SiO2
        ),
        # Three blocks of SiPD
        mp.Block(
            center=mp.Vector3(-design_region_width/3, -Sy/2 + PDsize/2, 0),
            size=mp.Vector3(subpixelsize, PDsize, 0),
            material=SiPD
        ),
        mp.Block(
            center=mp.Vector3(0, -Sy/2 + PDsize/2, 0),
            size=mp.Vector3(subpixelsize, PDsize, 0),
            material=SiPD
        ),
        mp.Block(
            center=mp.Vector3(design_region_width/3, -Sy/2 + PDsize/2, 0),
            size=mp.Vector3(subpixelsize, PDsize, 0),
            material=SiPD
        ),
    ]

    # Set up simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        default_material=Air,
        resolution=resolution,
        k_point=mp.Vector3(0, 0, 0),
    )

    # Define monitor positions and sizes
    monitor_position_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution)
    monitor_size_0 = mp.Vector3(subpixelsize, 0)

    monitor_position_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution)
    monitor_size_1 = mp.Vector3(subpixelsize, 0)

    monitor_position_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution)
    monitor_size_2 = mp.Vector3(subpixelsize, 0)

    # Create FourierFields for each position
    FourierFields_0 = mpa.FourierFields(sim, mp.Volume(center=monitor_position_0, size=monitor_size_0), mp.Ez, yee_grid=True)
    FourierFields_1 = mpa.FourierFields(sim, mp.Volume(center=monitor_position_1, size=monitor_size_1), mp.Ez, yee_grid=True)
    FourierFields_2 = mpa.FourierFields(sim, mp.Volume(center=monitor_position_2, size=monitor_size_2), mp.Ez, yee_grid=True)

    ob_list = [FourierFields_0, FourierFields_1, FourierFields_2]

    def J_0(fields_0, fields_1, fields_2):
        """
        Objective function that sums intensities in three frequency bands 
        corresponding to red, green, and blue.
        """
        red = npa.sum(npa.abs(fields_0[21:30, :]) ** 2)
        green = npa.sum(npa.abs(fields_1[11:20, :]) ** 2)
        blue = npa.sum(npa.abs(fields_2[1:10, :]) ** 2)

        # Ensure we handle AutoDiff types vs. raw floats
        red_ = red._value if hasattr(red, '_value') else red
        green_ = green._value if hasattr(green, '_value') else green
        blue_ = blue._value if hasattr(blue, '_value') else blue

        # Append and save
        red_list.append(red_)
        blue_list.append(blue_)
        green_list.append(green_)
        save_lists(red_list, blue_list, green_list)

        return blue + green + red

    # Define optimization problem
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J_0],
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies,
        decay_by=1e-3,  # field decay threshold
    )

    # Flatten and update design
    flattened_array = struct_np.flatten()
    opt.update_design([flattened_array])

    # Compute figure of merit and gradient
    fom, dJ_du = opt([flattened_array])
    g = npa.sum(dJ_du, axis=1)

    return fom, g

def waveguide_sim(struct_np, t, exp_name, prop_dir='top',
                  save_inter=False, interval=1, flag_last=False):
    """
    Waveguide simulation function:

    - Takes a 2D 'struct_np' array (clipped to 0/1).
    - Places it as a design region within a simple waveguide geometry in MEEP.
    - 'prop_dir' specifies which waveguide we are measuring transmission into
      ('top', 'bottom', or 'front').
    - Returns the figure of merit and its gradient.
    """

    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import matplotlib.pyplot as plt
    import wandb  # If not using wandb, remove or comment out.

    mp.verbosity(0)

    # Clip design values
    struct_np[struct_np > 1] = 1
    struct_np[struct_np < 0] = 0
    struct_np = np.squeeze(struct_np)

    # Define materials
    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)

    resolution = 21

    # Cell size
    Sx = 10
    Sy = 10
    cell_size = mp.Vector3(Sx, Sy)
    pml_layers = [mp.PML(2.0)]

    # Frequency setup
    fcen = 1 / 1.55
    width = 0.2
    fwidth = fcen * width

    # Source setup
    source_center = mp.Vector3(-2.7, 0, 0)
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
    Nx = 64
    Ny = 64

    # Define the design region
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), 
        SiO2, 
        Si, 
        grid_type="U_MEAN"
    )
    design_region = mpa.DesignRegion(
        design_variables, 
        volume=mp.Volume(
            center=mp.Vector3(), 
            size=mp.Vector3(3, 3, 0)
        )
    )

    # Geometry depending on 'prop_dir'
    if prop_dir == 'top':
        geometry = [
            # Horizontal waveguide on the left (origin)
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), 
                material=Si, 
                size=mp.Vector3(Sx / 2, 1, 0)
            ),
            # Vertical waveguide on the top
            mp.Block(
                center=mp.Vector3(y=Sy / 4), 
                material=Si, 
                size=mp.Vector3(1, Sy / 2, 0)
            ),
            # Design region
            mp.Block(
                center=design_region.center, 
                size=design_region.size, 
                material=design_variables
            ),
        ]
    elif prop_dir == 'bottom':
        geometry = [
            # Horizontal waveguide on the left (origin)
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), 
                material=Si, 
                size=mp.Vector3(Sx / 2, 1, 0)
            ),
            # Vertical waveguide on the bottom
            mp.Block(
                center=mp.Vector3(y=-Sy / 4), 
                material=Si, 
                size=mp.Vector3(1, Sy / 2, 0)
            ),
            # Design region
            mp.Block(
                center=design_region.center, 
                size=design_region.size, 
                material=design_variables
            ),
        ]
    elif prop_dir == 'front':
        geometry = [
            # Horizontal waveguide on the left
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), 
                material=Si, 
                size=mp.Vector3(Sx / 2, 1, 0)
            ),
            # Horizontal waveguide on the right ("front")
            mp.Block(
                center=mp.Vector3(x=Sx / 4), 
                material=Si, 
                size=mp.Vector3(Sx / 2, 1, 0)
            ),
            # Design region
            mp.Block(
                center=design_region.center, 
                size=design_region.size, 
                material=design_variables
            ),
        ]

    # Create the simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        eps_averaging=True,
        subpixel_tol=1e-4,
        resolution=resolution,
    )

    # Define mode monitors
    TE_front = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
    )
    TE_top = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)), mode=1
    )
    TE_bottom = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)), mode=1, forward=False
    )
    TE_o = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
    )

    ob_list = [TE_o]  # Always include the source monitor
    if prop_dir == 'top':
        ob_list.append(TE_top)
    elif prop_dir == 'bottom':
        ob_list.append(TE_bottom)
    elif prop_dir == 'front':
        ob_list.append(TE_front)

    def J(source_coeff, target_coeff):
        """
        Objective function: maximize transmission 
        = |target_coeff / source_coeff|^2
        """
        return npa.abs(target_coeff / source_coeff) ** 2

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
    )

    # Flatten and update design
    flattened_array = struct_np.flatten()
    opt.update_design([flattened_array])

    # Compute figure of merit (FOM) and gradient
    fom, g = opt([flattened_array])
    fom = fom[0]  # Single frequency -> single objective

    return fom, g
