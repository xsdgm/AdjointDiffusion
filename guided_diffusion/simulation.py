import matplotlib.pyplot as plt

import pickle
import os
# Define global lists


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5,3.5)

params = {
    'axes.labelsize':12, # label 字体大小
    'axes.titlesize':12, # 标题字体大小
    'xtick.labelsize':10, # x 轴刻度标签字体大小
    'ytick.labelsize':10, # y 轴刻度标签字体大小
    'xtick.direction': 'in', # 刻度显示方向 (in, out, inout)
    'ytick.direction': 'in', # 刻度显示方向 (in, out, inout)
    'lines.markersize': 3, # 标记点大小
    'axes.titlepad': 6, # 标题与图之间的间距
    'axes.labelpad': 4, # 坐标轴标签与图之间的间距
    'font.size': 12, # 字体大小
    #'font.sans-serif': 'Arial', # 字体设置
    'figure.dpi': 300, # 分辨率，vector 图形不受 dpi 影响也可清晰输出
    'figure.autolayout': True, # 自动布局（确保图中元素位于 figure 内部）
    'xtick.top': True, # 显示上侧 x 轴刻度
    'ytick.right': True, # 显示右侧 y 轴刻度
    'xtick.major.size': 2, # x 轴刻度长度
    'ytick.major.size': 2, # y 轴刻度长度
}

# Save the lists to a file

def save_lists(new_red_list, new_blue_list, new_green_list):
    # Load existing data if file exists
    if os.path.exists('lists.pkl'):
        with open('lists.pkl', 'rb') as f:
            red_list, blue_list, green_list = pickle.load(f)
    else:
        # If file does not exist, initialize empty lists
        red_list, blue_list, green_list = [], [], []

    # Append new data to existing lists
    red_list += new_red_list
    blue_list += new_blue_list
    green_list += new_green_list

    # Save the updated lists back to the pickle file
    with open('lists.pkl', 'wb') as f:
        pickle.dump((red_list, blue_list, green_list), f)


def CIS_sim(struct_np, t, exp_name, prop_dir ='',
                save_inter=False, interval=10, flag_last=False):
    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import numpy as npa
    import os
    import matplotlib.pyplot as plt
    struct_np [struct_np > 1] = 1
    struct_np [struct_np < 0 ] = 0

    # print(struct_np)
    red_list = []
    blue_list = []
    green_list = []

    mp.verbosity(0)
    Air = mp.Medium(index=1.0)
    SiN = mp.Medium(epsilon=4)
    SiO2 = mp.Medium(epsilon=2.1)
    SiPD = mp.Medium(epsilon=5)


    um_scale = 1
    resolution = 21

    design_region_width = 3 # 设计区域宽度
    design_region_height = 3 # 设计区域高度
    gapop = 0 ####################################################################################################
    air_gap = 0
    dti = 0.4
    subpixelsize = design_region_width/3 - dti
    if gapop == 1:
        air_gap = dti/2
    PDsize = 2
    Lpml = 0.5 # PML 区域大小
    Sourcespace = 2

    Sx = design_region_width
    Sy = PDsize + design_region_height + Sourcespace + Lpml
    cell_size = mp.Vector3(Sx, Sy)

    pml_layers = [mp.PML(thickness = Lpml, direction = mp.Y)]

    # 波长、频率设置
    wavelengths = np.linspace(0.40*um_scale, 0.70*um_scale, 31) 
    frequencies = 1/wavelengths
    nf = len(frequencies) # number of frequencies


    #source
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

    source_center = mp.Vector3(0, Sy/ 2 - Lpml - Sourcespace / 2, 0) # Source 位置
    source_size = mp.Vector3(Sx, 0, 0)

    source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center,),]

    Nx = 64#int(round(design_region_resolution * design_region_width)) + 1
    Ny = 64#int(round(design_region_resolution * design_region_height)) + 1

    # 基于设计区域与材料设置设计变量区域
    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, SiN, grid_type="U_MEAN")
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, - Sy /2 + PDsize + design_region_height / 2, 0),
            size=mp.Vector3(design_region_width, design_region_height, 0),
        ),
    )

    """
        
    """
    # 创建与 design region 相同尺寸的 Block
    geometry = [
        mp.Block(
            center=design_region.center, size=design_region.size, material=design_variables
        ),
        mp.Block(
            center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(Sx, PDsize, 0), material=SiO2
        ),
        # 有 DTI 时使用
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

    # Meep 仿真设置
    sim = mp.Simulation(
        cell_size=cell_size, 
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        default_material=Air, # 空白空间
        resolution=resolution,
        k_point = mp.Vector3(0,0,0) # bloch boundary
    )

    
    # 监视器位置和尺寸设置 (focal point)
    #monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
    #monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
    #monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0)
    #monitor_position_3, monitor_size_3 = mp.Vector3(0, -Sy/2 + PDsize + design_region_height + 0.5/resolution), mp.Vector3(design_region_width,0)
    # 监视器位置和尺寸设置 (focal point)
    monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
    monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
    monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0)



    # 通过 FourierFields 计算 monitor_position 处 monitor_size 区域的傅里叶变换
    FourierFields_0 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_0,size=monitor_size_0),mp.Ez,yee_grid=True)

    FourierFields_1 = mpa.FourierFields(sim,mp.Volume(center=monitor_position_1,size=monitor_size_1),mp.Ez,yee_grid=True)

    FourierFields_2= mpa.FourierFields(sim,mp.Volume(center=monitor_position_2,size=monitor_size_2),mp.Ez,yee_grid=True)


    ob_list = [FourierFields_0, FourierFields_1, FourierFields_2,]
        
    
    def J_0(fields_0, fields_1, fields_2):
        red = npa.sum(npa.abs(fields_0[21:30,:]) **2)
        green = npa.sum(npa.abs(fields_1[11:20,:]) ** 2) 
        blue = npa.sum(npa.abs(fields_2[1:10,:]) ** 2) 

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

        red_list.append(red_)
        blue_list.append(blue_)
        green_list.append(green_)
        save_lists(red_list, blue_list, green_list)

        return blue + green + red
    

    # 优化设置
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J_0],
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies,
        decay_by=1e-3, # 监视器中残余场值的比例
    )

    #struct_random = np.random.choice([0, 1], size=(Nx*Ny))

    flattened_array = struct_np.flatten()
    
    # error handling
    #try:  
    opt.update_design([flattened_array])
    #opt.plot2D(True)
    #except:
    #   exit(1)
    
    


    #opt.plot2D(fields=mp.Ez)
    #plt.savefig('structure.png')
    fom_arr, dJ_du = opt([flattened_array])
    # Safe robust aggregation regardless of whether wideband or single frequency is used
    g = np.array(dJ_du).reshape(-1, len(flattened_array)).sum(axis=0)
    fom = float(np.sum(fom_arr)) # or np.mean depending on objective scaling

    return fom, g




def waveguide_sim(struct_np, t, exp_name, prop_dir='top',
                save_inter=False, interval=1, flag_last=False, wideband=False):
    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import os
    import matplotlib.pyplot as plt
    
    assert prop_dir in ['top', 'bottom', 'front']
    mp.verbosity(0)
    struct_np [struct_np > 1] = 1
    struct_np [struct_np < 0 ] = 0
    struct_np = np.squeeze(struct_np)
    # print(struct_np)

    #mp.verbosity(0)
    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)

    resolution = 21

    Sx = 10
    Sy = 10
    cell_size = mp.Vector3(Sx, Sy)
    
    # pml_layers = [mp.PML(1.0)]
    pml_layers = [mp.PML(2.0)]
    
    fcen = 1 / 1.55
    width = 0.2
    fwidth = width * fcen

    if wideband:
        # Cover 1.53 ~ 1.57 μm
        nf_wideband = 7
        df_wideband = 1/1.53 - 1/1.57
    else:
        nf_wideband = 1
        df_wideband = 0
        
    # source_center = [-3.1, 0, 0]
    # source_size = mp.Vector3(0, 1, 0)
    # source_center = [-3.2, 0, 0]
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
        
    """
    if prop_dir == 'top':
        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),  # horizontal waveguide: left (origin)
            mp.Block(
                center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
            ),  # vertical waveguide: top
            mp.Block(
                center=design_region.center, size=design_region.size, material=design_variables
            ),  # design region
            # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
            # currently there is an issue of doing that; We give an alternative approach to impose symmetry in later tutorials.
            # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
        ]
        
    elif prop_dir == 'bottom':
        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),  # horizontal waveguide: left (origin)
            mp.Block(
                center=mp.Vector3(y=-Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
            ),  # vertical waveguide: bottom
            mp.Block(
                center=design_region.center, size=design_region.size, material=design_variables
            ),  # design region
            # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
            # currently there is an issue of doing that; We give an alternative approach to impose symmetry in later tutorials.
            # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
        ]
    elif prop_dir == 'front':
        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),  # horizontal waveguide: left (origin)
            mp.Block(
                center=mp.Vector3(x=Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),  # horizontal waveguide: front
            mp.Block(
                center=design_region.center, size=design_region.size, material=design_variables
            ),  # design region
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
    TE_bottom = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)), mode=1, forward=False
    )
    TE_o = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1
    )
    
    ob_list = [TE_o]
    
    if prop_dir == 'top':
        ob_list.append(TE_top)
        # ob_list = [TE_o, TE_top]
    elif prop_dir == 'bottom':
        ob_list.append(TE_bottom)
        # ob_list = [TE_o, TE_bottom]
    elif prop_dir == 'front':
        ob_list.append(TE_front)
        # ob_list = [TE_o, TE_front]
    
    def J(source, target):
        return npa.abs(target / source) ** 2
    
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        fcen=fcen,
        df=df_wideband,
        nf=nf_wideband,
    )

    #struct_random = np.random.choice([0, 1], size=(Nx*Ny))

    flattened_array = struct_np.flatten()
    opt.update_design([flattened_array])
    #opt.plot2D(True)
    
    #if save_inter and (t % interval == 0):
    #    os.makedirs(f'figures/{exp_name}', exist_ok=True)
    #    plt.savefig(f'figures/{exp_name}/structure_t={wandb.config.tsr-t}.png')
            
    #if flag_last:
    #    os.makedirs(f'figures/{exp_name}', exist_ok=True)
    #    plt.savefig(f'figures/{exp_name}/structure_final.png')


    # if filename is not None:
    #     plt.savefig(f'structure_{filename}.png')
    # else:
    #     plt.savefig('sample/structure.png')
    #opt.plot2D(fields=mp.Ez)
    #plt.savefig('structure.png')

    fom_arr, g_arr = opt([flattened_array])
    
    n_params = Nx * Ny
    if wideband:
        g = np.array(g_arr).reshape(-1, n_params).sum(axis=0)
        fom = float(np.mean(fom_arr))
    else:
        g = np.array(g_arr).flatten()
        fom = float(np.sum(fom_arr)) # or np.mean depending on array 
    
    return fom, g


def waveguide_sim_single(struct_np, t, exp_name, prop_dir='top',
                   save_inter=False, interval=1, flag_last=False):
    return waveguide_sim(struct_np, t, exp_name, prop_dir,
                   save_inter, interval, flag_last, wideband=False)


def waveguide_sim_wideband(struct_np, t, exp_name, prop_dir='top',
                     save_inter=False, interval=1, flag_last=False):
    return waveguide_sim(struct_np, t, exp_name, prop_dir,
                   save_inter, interval, flag_last, wideband=True)


def pbs_sim(struct_np, t, exp_name, prop_dir='top',
            save_inter=False, interval=1, flag_last=False, wideband=False):
    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa

    mp.verbosity(0)
    struct_np[struct_np > 1] = 1
    struct_np[struct_np < 0] = 0
    struct_np = np.squeeze(struct_np)

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

    # Wideband vs single-wavelength
    if wideband:
        # Cover 1.53 ~ 1.57 μm (7 frequency points) for better convergence
        nf_wideband = 7
        df_wideband = 1/1.53 - 1/1.57  # narrower frequency bandwidth
    else:
        # Single wavelength at 1.55 μm (original mode)
        nf_wideband = 1
        df_wideband = 0

    source_center = [-2.7, 0, 0]
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)

    design_region_resolution = 21
    Nx = 64
    Ny = 64

    # Output waveguide parameters
    y_offset = 0.8
    wg_width = 0.5

    def _safe_reset():
        try:
            mp.reset_meep()
        except Exception:
            pass

    allowed_props = {"top", "bottom", "front", "pbs"}
    if prop_dir not in allowed_props:
        raise ValueError(f"prop_dir must be one of {sorted(allowed_props)}")

    cross_weight = 0.5

    def _run_pol(pol):
        _safe_reset()
        parity = mp.ODD_Z if pol == "TE" else mp.EVEN_Z
        src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
        source = [
            mp.EigenModeSource(
                src,
                eig_parity=parity,
                eig_band=1,
                direction=mp.NO_DIRECTION,
                eig_kpoint=kpoint,
                size=source_size,
                center=source_center,
            )
        ]

        design_variables = mp.MaterialGrid(
            mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
        )
        design_region = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0)),
        )

        geometry = [
            mp.Block(
                center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)
            ),  # horizontal waveguide: left (input)
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
            ),  # horizontal waveguide: right top (output TE)
            mp.Block(
                center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)
            ),  # horizontal waveguide: right bottom (output TM)
            mp.Block(
                center=design_region.center, size=design_region.size, material=design_variables
            ),  # design region
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

        port_source = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)),
            mode=1,
            eig_parity=parity,
        )
        port_top = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1,
            eig_parity=parity,
        )
        port_bottom = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(2.5, -y_offset, 0), size=mp.Vector3(y=1.0)),
            mode=1,
            eig_parity=parity,
        )

        if prop_dir == "top":
            ob_list = [port_source, port_top]

            def J(source_coef, top_coef):
                denom = source_coef + 1e-12
                return npa.abs(top_coef / denom) ** 2

        elif prop_dir == "bottom":
            ob_list = [port_source, port_bottom]

            def J(source_coef, bottom_coef):
                denom = source_coef + 1e-12
                return npa.abs(bottom_coef / denom) ** 2

        else:
            # PBS objective: TE -> top, TM -> bottom, suppress cross coupling.
            desired_port = port_top if pol == "TE" else port_bottom
            cross_port = port_bottom if pol == "TE" else port_top
            ob_list = [port_source, desired_port, cross_port]

            def J(source_coef, desired_coef, cross_coef):
                denom = source_coef + 1e-12
                desired = npa.abs(desired_coef / denom) ** 2
                cross = npa.abs(cross_coef / denom) ** 2
                return desired - cross_weight * cross

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=[J],
            objective_arguments=ob_list,
            design_regions=[design_region],
            fcen=fcen,
            df=df_wideband,
            nf=nf_wideband,
        )

        flattened_array = struct_np.flatten()
        opt.update_design([flattened_array])
        fom_arr, dJ_du = opt([flattened_array])
        n_params = Nx * Ny
        if wideband:
            # dJ_du may have shape (nf, Nx*Ny), (1, nf, Nx*Ny), etc.
            # Reshape to (-1, Nx*Ny) and sum over all non-param axes → (Nx*Ny,)
            g = np.array(dJ_du).reshape(-1, n_params).sum(axis=0)
            fom_avg = np.mean(fom_arr)
        else:
            # Single frequency: dJ_du is 1D or (1, Nx*Ny)
            g = np.array(dJ_du).flatten()
            fom_avg = float(np.real(fom_arr).flatten()[0])
        return fom_avg, g

    _safe_reset()
    fom_te, g_te = _run_pol("TE")
    _safe_reset()
    fom_tm, g_tm = _run_pol("TM")

    fom = fom_te + fom_tm
    g = g_te + g_tm

    if flag_last:
        print(f"PBS final: fom_te={fom_te:.6f}, fom_tm={fom_tm:.6f}, fom_total={fom:.6f}")

    return fom, g


def pbs_sim_single(struct_np, t, exp_name, prop_dir='top',
                   save_inter=False, interval=1, flag_last=False):
    """PBS simulation with single wavelength (1.55 μm) — for adjoint-guided sampling."""
    return pbs_sim(struct_np, t, exp_name, prop_dir,
                   save_inter, interval, flag_last, wideband=False)


def pbs_sim_wideband(struct_np, t, exp_name, prop_dir='top',
                     save_inter=False, interval=1, flag_last=False):
    """PBS simulation with narrowed wideband (1.53-1.57 μm, 7 freq) — for SAC training."""
    return pbs_sim(struct_np, t, exp_name, prop_dir,
                   save_inter, interval, flag_last, wideband=True)