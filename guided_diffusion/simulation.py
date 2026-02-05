import matplotlib.pyplot as plt

import pickle
import os
# Define global lists


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5,3.5)

params = {
    'axes.labelsize':12, # label 폰트 크기
    'axes.titlesize':12, # 타이틀 폰트 크기
    'xtick.labelsize':10, # x 축 tick label 폰트 크기
    'ytick.labelsize':10, # y 축 tick label 폰트 크기 
    'xtick.direction': 'in', # 눈금 표시 방향 (in, out, inout)
    'ytick.direction': 'in', # 눈금 표시 방향 (in, out, inout)
    'lines.markersize': 3, # 마커 사이즈
    'axes.titlepad': 6, # 타이틀과 그래프 사이의 간격
    'axes.labelpad': 4, # 축 label과 그래프 사이의 간격
    'font.size': 12, # font 크기
    #'font.sans-serif': 'Arial', # font 설정
    'figure.dpi': 300, # 해상도, vector그래픽의 경우 dpi에 상관없이 깔끔하게 출력됨
    'figure.autolayout': True, # 레이아웃 자동 설정 (그래프의 모든 요소가 figure 내부에 들어가도록 설정)
    'xtick.top': True, # 그래프 위쪽 x축 눈금 표시
    'ytick.right': True, # 그래프 오른쪽 y축 눈금 표시
    'xtick.major.size': 2, # x축 눈금의 길이
    'ytick.major.size': 2, # y축 눈금의 길이
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
    import wandb
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

    design_region_width = 3 # 디자인 영역 너비
    design_region_height = 3 # 디자인 영역 높이
    gapop = 0 ####################################################################################################
    air_gap = 0
    dti = 0.4
    subpixelsize = design_region_width/3 - dti
    if gapop == 1:
        air_gap = dti/2
    PDsize = 2
    Lpml = 0.5 # PML 영역 크기
    Sourcespace = 2

    Sx = design_region_width
    Sy = PDsize + design_region_height + Sourcespace + Lpml
    cell_size = mp.Vector3(Sx, Sy)

    pml_layers = [mp.PML(thickness = Lpml, direction = mp.Y)]

    # 파장, 주파수 설정
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

    source_center = mp.Vector3(0, Sy/ 2 - Lpml - Sourcespace / 2, 0) # Source 위치
    source_size = mp.Vector3(Sx, 0, 0)

    source = [mp.Source(src_0, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_1, component=mp.Ez, size=source_size, center=source_center,),
            mp.Source(src_2, component=mp.Ez, size=source_size, center=source_center,),]

    Nx = 64#int(round(design_region_resolution * design_region_width)) + 1
    Ny = 64#int(round(design_region_resolution * design_region_height)) + 1

    # 설계 영역과 물질을 바탕으로 설계 영역 설정
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
    # design region과 동일한 size의 Block 생성
    geometry = [
        mp.Block(
            center=design_region.center, size=design_region.size, material=design_variables
        ),
        mp.Block(
            center=mp.Vector3(0, -Sy/2 + PDsize/2, 0), size=mp.Vector3(Sx, PDsize, 0), material=SiO2
        ),
        # DTI가 있을 경우 사용
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

    # Meep simulation 세팅
    sim = mp.Simulation(
        cell_size=cell_size, 
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=source,
        default_material=Air, # 빈공간
        resolution=resolution,
        k_point = mp.Vector3(0,0,0) # bloch boundary
    )

    
    # 모니터 위치와 크기 설정 (focal point)
    #monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
    #monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0) 
    #monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(0.01,0)
    #monitor_position_3, monitor_size_3 = mp.Vector3(0, -Sy/2 + PDsize + design_region_height + 0.5/resolution), mp.Vector3(design_region_width,0)
    #모니터 위치와 크기 설정 (focal point)
    monitor_position_0, monitor_size_0 = mp.Vector3(-design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
    monitor_position_1, monitor_size_1 = mp.Vector3(0, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0) 
    monitor_position_2, monitor_size_2 = mp.Vector3(design_region_width/3, -Sy/2 + PDsize - 0.5/resolution), mp.Vector3(subpixelsize,0)



    # FourierFields를 통해 monitor_position에서 monitor_size만큼의 영역에 대한 Fourier transform을 구함
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
    

    # optimization 설정å
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J_0],
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies,
        decay_by=1e-3, # 모니터에 남아있는 필드 값의 비율
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
    fom, dJ_du = opt([flattened_array])
    g = np.sum(dJ_du, axis=1)
    #f0 = f0[0]  
    #print(f0)
    #print(dJ_du)

    return fom, g




def waveguide_sim(struct_np, t, exp_name, prop_dir='top',
                save_inter=False, interval=1, flag_last=False):
    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import os
    import matplotlib.pyplot as plt
    import wandb
    
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
        df=0,
        nf=1,
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

    fom, g = opt([flattened_array])
    fom = fom[0]
    #f0 = f0[0]  
    #print(f0)
    #print(dJ_du)
    return fom, g


def pbs_sim(struct_np, t, exp_name, prop_dir='top',
            save_inter=False, interval=1, flag_last=False):
    import numpy as np
    import meep as mp
    import meep.adjoint as mpa
    import autograd.numpy as npa
    import wandb

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

    source_center = [-2.7, 0, 0]
    source_size = mp.Vector3(0, 2, 0)
    kpoint = mp.Vector3(1, 0, 0)

    design_region_resolution = 21
    Nx = 64
    Ny = 64

    cross_weight = 0.5

    def _safe_reset():
        try:
            mp.reset_meep()
        except Exception:
            pass

    def _run_pol(pol, prefer_dir):
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
            ),  # horizontal waveguide: left (origin)
            mp.Block(
                center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
            ),  # vertical waveguide: top
            mp.Block(
                center=mp.Vector3(y=-Sy / 4), material=Si, size=mp.Vector3(1, Sy / 2, 0)
            ),  # vertical waveguide: bottom
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
            mp.Volume(center=mp.Vector3(0, 2.5, 0), size=mp.Vector3(x=2)),
            mode=1,
            eig_parity=parity,
        )
        port_bottom = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(center=mp.Vector3(0, -2.5, 0), size=mp.Vector3(x=2)),
            mode=1,
            eig_parity=parity,
            forward=False,
        )

        ob_list = [port_source, port_top, port_bottom]

        def J(source_coef, top_coef, bottom_coef):
            denom = source_coef + 1e-12
            top_ratio = npa.abs(top_coef / denom) ** 2
            bottom_ratio = npa.abs(bottom_coef / denom) ** 2
            if prefer_dir == "top":
                return top_ratio - cross_weight * bottom_ratio
            return bottom_ratio - cross_weight * top_ratio

        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=J,
            objective_arguments=ob_list,
            design_regions=[design_region],
            fcen=fcen,
            df=0,
            nf=1,
        )

        flattened_array = struct_np.flatten()
        opt.update_design([flattened_array])
        fom, g = opt([flattened_array])
        return fom[0], g

    _safe_reset()
    fom_te, g_te = _run_pol("TE", "top")
    _safe_reset()
    fom_tm, g_tm = _run_pol("TM", "bottom")

    fom = fom_te + fom_tm
    g = g_te + g_tm

    if flag_last:
        wandb.log({"fom_te": fom_te, "fom_tm": fom_tm, "fom_pbs": fom})

    return fom, g