import os
import random
import nlopt
import numpy as np
import matplotlib.pyplot as plt

import meep as mp
import meep.adjoint as mpa
from scipy.ndimage import label, binary_dilation
from skimage.measure import euler_number
from skimage.measure import label as label_
from autograd import numpy as npa
from autograd import tensor_jacobian_product

# =============================================================================
# 初始化路径
# =============================================================================
directory = 'logs'
if not os.path.exists(directory):
    os.makedirs(directory)

# Matplotlib 配置
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["figure.figsize"] = (3.5, 3.5)
plt.rcParams.update({
    'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.markersize': 3, 'axes.titlepad': 6, 'axes.labelpad': 4,
    'font.size': 12, 'figure.dpi': 300, 'figure.autolayout': True,
    'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 2, 'ytick.major.size': 2
})

# =============================================================================
# 基础常量设定 (与 pbs_eval.py 保持一致)
# =============================================================================
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

Nx = 64
Ny = 64
cross_weight = 0.5   # 抑制串扰的权重
y_offset = 0.8
wg_width = 0.5

# 设计区域相关设置
design_region_resolution = 21
design_region_width = 3
design_region_height = 3



mp.verbosity(0)
# 初始化两组独立的目标函数优化器
print("Initializing Optimization Environments for TE and TM...")

# =============================================================================
# 优化器及过滤/投影设置
# =============================================================================
minimum_length = 0.895
eta_i = 0.5
eta_e = 0.75
filter_radius = minimum_length



def mapping(x, eta, beta):
    """ 对特征进行平滑和投影映射 """
    x_filtered = mpa.conic_filter(
        x, filter_radius, design_region_width, design_region_height, design_region_resolution
    )
    projected_field = mpa.tanh_projection(x_filtered, beta, eta)
    return projected_field.flatten()

# 初始设计参数
x = 0.5 * np.ones((Nx * Ny,))
evaluation_history = []
k = 0

def run_single_polarization(design_array, pol: str):
    """ 运行单次极化仿真并返回FOM和梯度 """
    try:
        mp.reset_meep()
    except Exception:
        pass
    
    # 为每个偏振创建独立的设计变量
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
    )
    
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_height, 0))
    )
    
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

    geometry = [
        mp.Block(center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)),
        mp.Block(center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)),
        mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
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
        sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1, eig_parity=parity
    )
    port_top = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(2.5, y_offset, 0), size=mp.Vector3(y=1.0)), mode=1, eig_parity=parity
    )
    port_bottom = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(2.5, -y_offset, 0), size=mp.Vector3(y=1.0)), mode=1, eig_parity=parity
    )

    ob_list = [port_source, port_top, port_bottom]

    def J_top(src_coef, top_coef, bot_coef):
        power_source = npa.abs(src_coef)**2 + 1e-12
        trans_top = npa.abs(top_coef)**2 / power_source
        trans_bot = npa.abs(bot_coef)**2 / power_source
        return trans_top - cross_weight * trans_bot

    def J_bottom(src_coef, top_coef, bot_coef):
        power_source = npa.abs(src_coef)**2 + 1e-12
        trans_top = npa.abs(top_coef)**2 / power_source
        trans_bot = npa.abs(bot_coef)**2 / power_source
        return trans_bot - cross_weight * trans_top

    if pol == "TE":
        opt_func = [J_top]
    else:
        opt_func = [J_bottom]

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=opt_func,
        objective_arguments=ob_list,
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
    )
    
    # 更新设计变量
    opt.update_design([design_array])
    
    # 运行仿真并获取结果
    fom, g = opt([design_array])
    
    return fom, g

def f(x, grad, beta):
    """ Nlopt回调函数：评估FOM并计算梯度 """
    global k
    x_mapped = mapping(x, eta_i, beta)
    
    # 分别运行TE和TM的伴随仿真获取梯度
    f0_te, dJ_du_te = run_single_polarization(x_mapped, "TE")
    f0_tm, dJ_du_tm = run_single_polarization(x_mapped, "TM")
    
    # 取单一频率的标量结果
    f0_te_val = f0_te[0]
    f0_tm_val = f0_tm[0]
    
    # PBS 总目标函数及总梯度
    f0_total = f0_te_val + f0_tm_val
    dJ_du_total = dJ_du_te + dJ_du_tm
    
    # 如果 nlopt 请求梯度，则根据投影函数应用链式法则
    if grad.size > 0:
        grad[:] = tensor_jacobian_product(mapping, 0)(x, eta_i, beta, dJ_du_total)
    
    # 计算用于记录的梯度范数
    adjgrad_norm = np.linalg.norm(dJ_du_total.reshape(dJ_du_total.shape[0], -1), axis=1).mean()
    
    print(f'Step {k:03d} | FOM Total: {f0_total:.4f} (TE: {f0_te_val:.4f}, TM: {f0_tm_val:.4f}) | Grad Norm: {adjgrad_norm:.4f}')
    
    evaluation_history.append(np.real(f0_total))
    
    # 存图与数据
    filename_prefix = f'{directory}/{str(k).zfill(2)}_{str(np.real(f0_total))[:5]}'
    
    # 我们调用 opt_TE 的绘图即可，因为共享一个设计区域
    opt_TE.update_design([x_mapped])
    opt_TE.plot2D()
    plt.savefig(filename_prefix + '.png')
    np.save(filename_prefix + '.npy', x)
    plt.close()
    
    k += 1
    return np.real(f0_total)

# =============================================================================
# NLopt 优化流程
# =============================================================================
algorithm = nlopt.LD_MMA
n = Nx * Ny
cur_beta = 2
beta_scale = 2
num_betas = 7
update_factor = 20

for iters in range(num_betas):
    print(f"\n--- Starting Iteration {iters+1}/{num_betas} | Current beta: {cur_beta} ---")
    
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    
    if cur_beta >= 2 ** (num_betas + 1):
        solver.set_max_objective(lambda a, g: f(a, g, mp.inf))
        solver.set_maxeval(1)
    else:
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor)
    
    np.savetxt(f'{directory}/history.txt', evaluation_history)
    x[x > 1] = 1
    x[x < 0] = 0
    x = solver.optimize(x)
    
    cur_beta *= beta_scale

# =============================================================================
# 优化结束：最终结构评估和后处理
# =============================================================================
x_final = mapping(x, eta_i, mp.inf)

# 创建一个新的优化问题对象用于最终评估
try:
    mp.reset_meep()
except Exception:
    pass

# 创建设计变量和设计区域
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN"
)

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_height, 0))
)

# 创建仿真对象
parity = mp.ODD_Z  # TE 模式
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

geometry = [
    mp.Block(center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 1, 0)),
    mp.Block(center=mp.Vector3(x=Sx / 4, y=y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)),
    mp.Block(center=mp.Vector3(x=Sx / 4, y=-y_offset), material=Si, size=mp.Vector3(Sx / 2, wg_width, 0)),
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables),
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

# 创建优化问题对象
port_source = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(-2.5, 0, 0), size=mp.Vector3(y=2)), mode=1, eig_parity=parity
)
port_top = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(2.5, y_offset, 0), size=mp.Vector3(y=1.0)), mode=1, eig_parity=parity
)
port_bottom = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(2.5, -y_offset, 0), size=mp.Vector3(y=1.0)), mode=1, eig_parity=parity
)

ob_list = [port_source, port_top, port_bottom]

def J_top(src_coef, top_coef, bot_coef):
    power_source = npa.abs(src_coef)**2 + 1e-12
    trans_top = npa.abs(top_coef)**2 / power_source
    trans_bot = npa.abs(bot_coef)**2 / power_source
    return trans_top - cross_weight * trans_bot

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J_top],
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
)

# 更新设计变量并绘制结果
opt.update_design([x_final])
opt.plot2D()
plt.savefig(f"{directory}/final_structure.png")
np.save(f"{directory}/final.npy", x_final)

# 绘制FOM迭代历史
plt.figure()
plt.plot(np.array(evaluation_history), "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("PBS FOM")
plt.savefig(f'{directory}/log.png')

print("Optimization Finished Successfully.")