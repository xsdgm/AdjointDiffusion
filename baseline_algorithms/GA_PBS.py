import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from scipy.ndimage import label, binary_dilation
from skimage.measure import euler_number
from skimage.measure import label as label_

# ==========================================
# 初始化本地输出目录
# ==========================================
directory = 'logs'
img_directory = os.path.join(directory, 'images')

if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(img_directory):
    os.makedirs(img_directory)

# ==========================================
# 辅助函数：去除孤岛结构
# ==========================================
def delete_islands_with_size_1(array):
    labeled_array, num_features = label(array)
    islands_to_delete = [i for i in range(1, num_features + 1) if np.sum(labeled_array == i) == 1]
    for island in islands_to_delete:
        array[labeled_array == island] = 0
    if array.ndim > 2:
        array = np.squeeze(array, axis=2)
    return array

# ==========================================
# 仿真全局参数
# ==========================================
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

Nx = 64
Ny = 64
cross_weight = 0.5  # 交叉串扰权重

minimum_length = 0.895
eta_i = 0.5
design_region_width = 3
design_region_height = 3
design_region_resolution = 21
filter_radius = minimum_length

print(f"--- 仿真参数配置 ---")
print(f"最小线宽 (minimum length): {minimum_length}")
print(f"分辨率 (resolution): {resolution}")
print(f"网格: {Nx}x{Ny}, eta_i: {eta_i}, 串扰权重 (cross_weight): {cross_weight}")
print(f"--------------------\n")

# ==========================================
# 映射函数 (滤波与投影)
# ==========================================
def mapping(x, eta, beta):
    x = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width, 
        design_region_height,
        design_region_resolution
    )
    x = x.flatten()
    beta = float(beta)
    eta = float(eta)
    projected_field = mpa.tanh_projection(x, beta, eta)
    return projected_field.flatten()

# ==========================================
# 核心函数：单次极化仿真 (TE 或 TM)
# ==========================================
def run_single_polarization(design_array, pol: str):
    try:
        mp.reset_meep()
    except Exception:
        pass

    y_offset = 0.8
    wg_width = 0.5
    
    # 根据极化方向设定 parity
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

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3, 3, 0)),
    )

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

    def J_top(source_coef, top_coef, bottom_coef):
        denom = source_coef + 1e-12
        return npa.abs(top_coef / denom) ** 2

    def J_bottom(source_coef, top_coef, bottom_coef):
        denom = source_coef + 1e-12
        return npa.abs(bottom_coef / denom) ** 2

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J_top, J_bottom],
        objective_arguments=[port_source, port_top, port_bottom],
        design_regions=[design_region],
        fcen=fcen,
        df=0,
        nf=1,
    )

    opt.update_design([design_array])
    fom, g = opt([design_array])
    
    # 提取 FOM 与梯度
    fom_top = float(np.asarray(fom[0]).item())
    fom_bot = float(np.asarray(fom[1]).item())
    g_top = g[0]
    g_bot = g[1]
    
    return fom_top, fom_bot, g_top, g_bot

# ==========================================
# 初始化优化参数
# ==========================================
n = Nx * Ny
LR = 1
cur_beta = 2
beta_scale = 2
num_betas = 100

print(f"--- 优化参数配置 ---")
print(f"beta_scale: {beta_scale}, 迭代次数 (num_betas): {num_betas}, 基础学习率 (LR): {LR}")
print(f"--------------------\n")

# 初始均匀结构分布
x = np.ones(n) * 0.5

# 存储 FOM 值的列表
fom_history = []

# ==========================================
# 梯度上升主循环
# ==========================================
for iters in range(num_betas):
    x[x > 1] = 1
    x[x < 0] = 0
    
    # 进行映射
    x_mapped = mapping(x, eta_i, cur_beta)
    
    # 1. 计算 TE 模式
    fom_te_top, fom_te_bot, g_te_top, g_te_bot = run_single_polarization(x_mapped, "TE")
    
    # 2. 计算 TM 模式
    fom_tm_top, fom_tm_bot, g_tm_top, g_tm_bot = run_single_polarization(x_mapped, "TM")
    
    # 3. 合并 PBS FOM 与 梯度
    fom_pbs = (fom_te_top - cross_weight * fom_te_bot) + (fom_tm_bot - cross_weight * fom_tm_top)
    adjoint_gradient = (g_te_top - cross_weight * g_te_bot) + (g_tm_bot - cross_weight * g_tm_top)
    
    # 计算梯度范数以动态调整步长
    reshaped_gradients = adjoint_gradient.reshape(-1, 1)
    adjgrad_norm = np.linalg.norm(reshaped_gradients)
    learning_rate = LR / adjgrad_norm
    
    # 保存 FOM 值
    fom_history.append({
        'iteration': iters,
        'fom_pbs': fom_pbs,
        'fom_te_top': fom_te_top,
        'fom_te_bot': fom_te_bot,
        'fom_tm_top': fom_tm_top,
        'fom_tm_bot': fom_tm_bot,
        'learning_rate': learning_rate
    })
    
    # 打印终端日志
    print(f"Iter: {iters:03d} | PBS FOM: {fom_pbs:.4f} | TE_Top: {fom_te_top:.4f} | TM_Bot: {fom_tm_bot:.4f} | LR: {learning_rate:.4e}")
    
    # 每隔 10 次迭代，或是最后一次迭代，将生成的结构保存到本地
    if iters % 10 == 0 or iters == num_betas - 1:
        plt.figure(figsize=(4, 4))
        plt.imshow(x.reshape(Nx, Ny), cmap='gray')
        plt.title(f"Iteration {iters}")
        plt.colorbar()
        plt.savefig(os.path.join(img_directory, f"iter_{iters:03d}.png"))
        plt.close()
    
    # 参数更新 (梯度上升)
    x = x + learning_rate * adjoint_gradient
    cur_beta = cur_beta * beta_scale

# ==========================================
# 最终后处理及导出
# ==========================================
print(f"\nOptimization Finished. Final PBS FOM: {fom_pbs:.4f}")

# 二值化最终结构并保存图片
x_final = mapping(x, eta_i, mp.inf)
plt.figure(figsize=(4, 4))
plt.imshow(x_final.reshape(Nx, Ny), cmap='gray')
plt.title("Generated Final")
plt.colorbar()
plt.savefig(os.path.join(directory, "generated_final.png"))
plt.close()

# 去除孤岛结构并保存图片
x_final_2d = x_final.reshape(Nx, Ny)
island_deleted = delete_islands_with_size_1(x_final_2d.copy())

plt.figure(figsize=(4, 4))
plt.imshow(1 - island_deleted, cmap='gray')  # 翻转颜色，使得介质通常为黑色方便查看
plt.title("Generated Final (Island Deleted)")
plt.colorbar()
plt.savefig(os.path.join(directory, "generated_final_island_deleted.png"))
plt.close()

# 保存最终的 numpy 数组用于后续评估
np.save(os.path.join(directory, "pbs_final_struct.npy"), island_deleted)

# 保存 FOM 历史记录为 CSV 文件
csv_file_path = os.path.join(directory, "fom_history.csv")
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['iteration', 'fom_pbs', 'fom_te_top', 'fom_te_bot', 'fom_tm_top', 'fom_tm_bot', 'learning_rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in fom_history:
        writer.writerow(entry)
print(f"FOM history saved to CSV: {csv_file_path}")

print(f"Final structures saved to {directory}")
print("PBS Optimization DONE")