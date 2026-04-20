import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.transform import resize

from .pbs_platform import get_pbs_platform_config


def _vec3(mp, x=0.0, y=0.0, z=0.0):
    return mp.Vector3(x, y, z)


def wavelength_um_from_fcen(fcen):
    return 1.0 / float(fcen)


def index_with_linear_dispersion(index_ref, slope, wavelength_um, ref_wavelength_um):
    return float(index_ref + slope * (wavelength_um - ref_wavelength_um))


def db_per_cm_to_d_conductivity(loss_db_per_cm, wavelength_um, index_real):
    if loss_db_per_cm <= 0:
        return 0.0
    alpha_np_per_um = loss_db_per_cm * np.log(10.0) / (20.0 * 1.0e4)
    kappa = alpha_np_per_um * wavelength_um / (2.0 * np.pi)
    return float((4.0 * np.pi * index_real * kappa) / max(wavelength_um, 1e-12))


def normalize_vector(vec):
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        raise ValueError("Optic axis must have non-zero norm.")
    return arr / norm


def get_optic_axis(cfg):
    if cfg.optic_axis is not None:
        return normalize_vector(cfg.optic_axis)
    cut = (cfg.crystal_cut or "z-cut").lower()
    if cut == "x-cut":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if cut == "y-cut":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


def uniaxial_epsilon_tensor(no, ne, optic_axis):
    optic_axis = normalize_vector(optic_axis)
    base = (no ** 2) * np.eye(3)
    delta = (ne ** 2 - no ** 2) * np.outer(optic_axis, optic_axis)
    return base + delta


def tensor_to_meep_vectors(mp, tensor):
    tensor = np.asarray(tensor, dtype=float)
    diag = _vec3(mp, tensor[0, 0], tensor[1, 1], tensor[2, 2])
    offdiag = _vec3(mp, tensor[0, 1], tensor[0, 2], tensor[1, 2])
    return diag, offdiag


def conductivity_tensor(loss):
    if loss <= 0:
        return np.zeros((3, 3), dtype=float)
    return np.eye(3) * float(loss)


def get_objective_wavelengths(cfg):
    wavelengths = cfg.objective_wavelengths_um or (cfg.wavelength_um,)
    weights = cfg.objective_wavelength_weights or tuple(1.0 for _ in wavelengths)
    if len(wavelengths) != len(weights):
        raise ValueError(
            "objective_wavelengths_um and objective_wavelength_weights must have the same length."
        )
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return tuple(float(w) for w in wavelengths), tuple(float(w) for w in weights)


def get_tfln_material_model(cfg, wavelength_um):
    no = index_with_linear_dispersion(
        cfg.ordinary_index_ref,
        cfg.ordinary_dispersion_slope,
        wavelength_um,
        cfg.dispersion_ref_wavelength_um,
    )
    ne = index_with_linear_dispersion(
        cfg.extraordinary_index_ref,
        cfg.extraordinary_dispersion_slope,
        wavelength_um,
        cfg.dispersion_ref_wavelength_um,
    )
    clad = index_with_linear_dispersion(
        cfg.cladding_index_ref or cfg.cladding_index,
        cfg.cladding_dispersion_slope,
        wavelength_um,
        cfg.dispersion_ref_wavelength_um,
    )
    return no, ne, clad


def create_pbs_mediums(mp, cfg, fcen):
    wavelength_um = wavelength_um_from_fcen(fcen)
    if cfg.simulation_dim == 3 and cfg.ordinary_index_ref is not None and cfg.extraordinary_index_ref is not None:
        no, ne, clad = get_tfln_material_model(cfg, wavelength_um)
        loss = db_per_cm_to_d_conductivity(cfg.loss_db_per_cm, wavelength_um, no)
        epsilon_tensor = uniaxial_epsilon_tensor(no, ne, get_optic_axis(cfg))
        eps_diag, eps_offdiag = tensor_to_meep_vectors(mp, epsilon_tensor)
        sigma_diag, sigma_offdiag = tensor_to_meep_vectors(mp, conductivity_tensor(loss))
        core = mp.Medium(
            epsilon_diag=eps_diag,
            epsilon_offdiag=eps_offdiag,
            D_conductivity_diag=sigma_diag,
            D_conductivity_offdiag=sigma_offdiag,
        )
        top_cladding = mp.Medium(index=clad)
        bottom_cladding = mp.Medium(index=clad)
        return core, top_cladding, bottom_cladding
    if cfg.meep_material_name:
        try:
            meep_materials = __import__("meep.materials", fromlist=[cfg.meep_material_name])
            core = getattr(meep_materials, cfg.meep_material_name)
            clad = index_with_linear_dispersion(
                cfg.cladding_index_ref or cfg.cladding_index,
                cfg.cladding_dispersion_slope,
                wavelength_um,
                cfg.dispersion_ref_wavelength_um,
            )
            top_cladding = mp.Medium(index=clad)
            bottom_cladding = mp.Medium(index=clad)
            return core, top_cladding, bottom_cladding
        except Exception:
            pass
    if cfg.core_epsilon_diag is not None:
        core = mp.Medium(epsilon_diag=_vec3(mp, *cfg.core_epsilon_diag))
    else:
        core = mp.Medium(index=cfg.core_index)
    cladding = mp.Medium(index=cfg.cladding_index)
    return core, cladding, cladding


def get_pbs_mode_parity(mp, cfg, pol):
    if cfg.simulation_dim == 2:
        return mp.ODD_Z if pol == "TE" else mp.EVEN_Z
    even_y = getattr(mp, "EVEN_Y", 0)
    return even_y + (mp.ODD_Z if pol == "TE" else mp.EVEN_Z)


def get_pbs_dominant_component(mp, cfg, pol):
    field_name = cfg.dominant_field_te if pol == "TE" else cfg.dominant_field_tm
    return getattr(mp, field_name)


def make_design_grid(mp, cfg):
    if cfg.simulation_dim == 3:
        return _vec3(mp, cfg.grid_nx, cfg.grid_ny, cfg.grid_nz)
    return _vec3(mp, cfg.grid_nx, cfg.grid_ny)


def get_design_pixel_size_um(cfg):
    return cfg.design_region_size_um / cfg.grid_ny


def get_design_pixel_centers(cfg):
    pixel_size_um = get_design_pixel_size_um(cfg)
    half_size_um = 0.5 * cfg.design_region_size_um
    return np.linspace(
        -half_size_um + 0.5 * pixel_size_um,
        half_size_um - 0.5 * pixel_size_um,
        cfg.grid_ny,
    )


def get_design_pixel_boundaries(cfg):
    pixel_size_um = get_design_pixel_size_um(cfg)
    half_size_um = 0.5 * cfg.design_region_size_um
    return np.linspace(-half_size_um, half_size_um, cfg.grid_ny + 1)


def snap_coordinate_to_design_grid(value, cfg, location="center"):
    candidates = (
        get_design_pixel_centers(cfg)
        if location == "center"
        else get_design_pixel_boundaries(cfg)
    )
    candidates = np.asarray(candidates, dtype=float)
    return float(candidates[np.argmin(np.abs(candidates - value))])


def snap_length_to_design_pixels(length_um, cfg, center_location):
    pixel_size_um = get_design_pixel_size_um(cfg)
    pixel_count = max(int(round(length_um / pixel_size_um)), 1)

    # If the waveguide/monitor center sits on a pixel center, odd pixel counts
    # keep the edges on pixel boundaries. If it sits on a pixel boundary,
    # even pixel counts do the same.
    if center_location == "center" and pixel_count % 2 == 0:
        lower = max(pixel_count - 1, 1)
        upper = pixel_count + 1
        pixel_count = lower if abs(lower * pixel_size_um - length_um) <= abs(upper * pixel_size_um - length_um) else upper
    if center_location == "boundary" and pixel_count % 2 == 1:
        lower = max(pixel_count - 1, 2)
        upper = pixel_count + 1
        pixel_count = lower if abs(lower * pixel_size_um - length_um) <= abs(upper * pixel_size_um - length_um) else upper

    return float(pixel_count * pixel_size_um)


def get_transverse_port_layout(cfg):
    layout = {
        "source_span_y": float(cfg.source_span_y),
        "input_center_y": 0.0,
        "input_width_y": float(cfg.input_wg_width),
        "input_monitor_span_y": float(cfg.monitor_span_y_in),
        "output_center_y": float(cfg.output_offset_y),
        "output_width_y": float(cfg.output_wg_width),
        "output_monitor_span_y": float(cfg.monitor_span_y_out),
    }
    if not cfg.align_waveguides_to_pixels:
        return layout

    layout["source_span_y"] = snap_length_to_design_pixels(
        cfg.source_span_y, cfg, center_location="boundary"
    )
    layout["input_width_y"] = snap_length_to_design_pixels(
        cfg.input_wg_width, cfg, center_location="boundary"
    )
    layout["input_monitor_span_y"] = snap_length_to_design_pixels(
        cfg.monitor_span_y_in, cfg, center_location="boundary"
    )
    layout["output_center_y"] = snap_coordinate_to_design_grid(
        cfg.output_offset_y, cfg, location="center"
    )
    layout["output_width_y"] = snap_length_to_design_pixels(
        cfg.output_wg_width, cfg, center_location="center"
    )
    layout["output_monitor_span_y"] = snap_length_to_design_pixels(
        cfg.monitor_span_y_out, cfg, center_location="center"
    )
    return layout


def flatten_design_weights(struct, cfg):
    if cfg.simulation_dim == 3:
        return build_3d_process_weights(struct, cfg).flatten()
    struct = resize_structure_to_design_grid(struct, cfg)
    return struct.flatten()


def resize_structure_to_design_grid(struct, cfg):
    struct = np.asarray(struct, dtype=float)
    struct = np.squeeze(struct)
    if struct.shape == (cfg.grid_nx, cfg.grid_ny):
        return np.clip(struct, 0.0, 1.0)
    return np.clip(
        resize(
            struct,
            (cfg.grid_nx, cfg.grid_ny),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        ),
        0.0,
        1.0,
    )


def resize_gradient_to_structure_grid(grad, target_shape):
    grad = np.asarray(grad, dtype=float)
    grad = np.squeeze(grad)
    if grad.shape == tuple(target_shape):
        return grad
    return resize(
        grad,
        tuple(target_shape),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )


def project_design_gradient_to_structure(grad, cfg, target_shape):
    grad = np.asarray(grad, dtype=float)
    grad = np.squeeze(grad)
    if cfg.simulation_dim == 3:
        grad = grad.reshape(cfg.grid_nx, cfg.grid_ny, cfg.grid_nz).sum(axis=2)
    else:
        grad = grad.reshape(cfg.grid_nx, cfg.grid_ny)
    return resize_gradient_to_structure_grid(grad, target_shape)


def build_3d_process_weights(struct, cfg):
    struct_2d = resize_structure_to_design_grid(struct, cfg)
    binary = struct_2d > 0.5
    weights = np.zeros((cfg.grid_nx, cfg.grid_ny, cfg.grid_nz), dtype=float)
    if cfg.grid_nz <= 1:
        weights[:, :, 0] = struct_2d
        return weights

    slice_thickness = cfg.design_region_thickness_um / cfg.grid_nz
    pixel_size_um = cfg.design_region_size_um / cfg.grid_nx
    for iz in range(cfg.grid_nz):
        depth_from_top = iz * slice_thickness
        lateral_bias_um = depth_from_top * np.tan(np.deg2rad(cfg.sidewall_angle_deg))
        radius_px = int(round(lateral_bias_um / max(pixel_size_um, 1e-12)))
        prof = binary_dilation(binary, disk(radius_px)) if radius_px > 0 else binary
        weights[:, :, iz] = prof.astype(float)
    return weights


def get_cell_size(mp, cfg):
    return _vec3(mp, cfg.cell_size_x, cfg.cell_size_y, cfg.cell_size_z)


def get_film_bottom_z(cfg):
    return -0.5 * cfg.film_thickness_um


def get_slab_center_z(cfg):
    return get_film_bottom_z(cfg) + 0.5 * cfg.slab_thickness_um


def get_ridge_bottom_z(cfg):
    return get_film_bottom_z(cfg) + cfg.slab_thickness_um


def get_ridge_center_z(cfg):
    return get_ridge_bottom_z(cfg) + 0.5 * cfg.design_region_thickness_um


def get_design_volume(mp, cfg):
    return mp.Volume(
        center=_vec3(mp, z=get_ridge_center_z(cfg)),
        size=_vec3(
            mp,
            cfg.design_region_size_um,
            cfg.design_region_size_um,
            cfg.design_region_thickness_um,
        ),
    )


def get_source_size(mp, cfg):
    layout = get_transverse_port_layout(cfg)
    return _vec3(mp, 0, layout["source_span_y"], cfg.source_span_z)


def get_monitor_volume(mp, cfg, x, y, span_y):
    return mp.Volume(
        center=_vec3(mp, x, y, 0),
        size=_vec3(mp, 0, span_y, cfg.monitor_span_z),
    )


def build_pbs_geometry(mp, design_material, cfg, core):
    sx = cfg.cell_size_x
    sy = cfg.cell_size_y
    layout = get_transverse_port_layout(cfg)
    y_offset = layout["output_center_y"]
    wg_thickness = cfg.waveguide_thickness_um
    geometry = []
    if cfg.simulation_dim == 3 and cfg.slab_thickness_um > 0:
        geometry.append(
            mp.Block(
                center=_vec3(
                    mp,
                    z=get_slab_center_z(cfg),
                ),
                material=core,
                size=_vec3(mp, sx, sy, cfg.slab_thickness_um),
            )
        )
        geometry.extend(
            build_tapered_ridge_blocks(
                mp, cfg, core, center_x=-sx / 4, center_y=layout["input_center_y"], width=layout["input_width_y"], length=sx / 2
            )
        )
        geometry.extend(
            build_tapered_ridge_blocks(
                mp, cfg, core, center_x=sx / 4, center_y=y_offset, width=layout["output_width_y"], length=sx / 2
            )
        )
        geometry.extend(
            build_tapered_ridge_blocks(
                mp, cfg, core, center_x=sx / 4, center_y=-y_offset, width=layout["output_width_y"], length=sx / 2
            )
        )
    else:
        geometry.extend(
            [
                mp.Block(
                    center=_vec3(mp, x=-sx / 4, y=layout["input_center_y"]),
                    material=core,
                    size=_vec3(mp, sx / 2, layout["input_width_y"], wg_thickness),
                ),
                mp.Block(
                    center=_vec3(mp, x=sx / 4, y=y_offset),
                    material=core,
                    size=_vec3(mp, sx / 2, layout["output_width_y"], wg_thickness),
                ),
                mp.Block(
                    center=_vec3(mp, x=sx / 4, y=-y_offset),
                    material=core,
                    size=_vec3(mp, sx / 2, layout["output_width_y"], wg_thickness),
                ),
            ]
        )
    geometry.append(
        mp.Block(
            center=_vec3(mp, z=get_ridge_center_z(cfg)),
            size=_vec3(
                mp,
                cfg.design_region_size_um,
                cfg.design_region_size_um,
                cfg.design_region_thickness_um,
            ),
            material=design_material,
        )
    )
    return geometry


def build_tapered_ridge_blocks(mp, cfg, core, center_x, center_y, width, length):
    if cfg.grid_nz <= 1:
        return [
            mp.Block(
                center=_vec3(mp, center_x, center_y, get_ridge_center_z(cfg)),
                material=core,
                size=_vec3(mp, length, width, cfg.waveguide_thickness_um),
            )
        ]
    pixel_size_um = cfg.design_region_size_um / cfg.grid_nx
    slice_thickness = cfg.waveguide_thickness_um / cfg.grid_nz
    blocks = []
    for iz in range(cfg.grid_nz):
        depth_from_top = iz * slice_thickness
        lateral_bias_um = depth_from_top * np.tan(np.deg2rad(cfg.sidewall_angle_deg))
        expanded_width = width + 2.0 * max(lateral_bias_um, pixel_size_um * 0.0)
        z_bottom = get_ridge_bottom_z(cfg)
        z_center = z_bottom + cfg.waveguide_thickness_um / 2.0 - (iz + 0.5) * slice_thickness
        blocks.append(
            mp.Block(
                center=_vec3(mp, center_x, center_y, z_center),
                material=core,
                size=_vec3(mp, length, expanded_width, slice_thickness),
            )
        )
    return blocks


def build_pbs_simulation(mp, mpa, struct, pol, platform="soi", fcen=None, fwidth=None):
    cfg = get_pbs_platform_config(platform)
    fcen = (1 / cfg.wavelength_um) if fcen is None else fcen
    fwidth = (cfg.source_width * fcen) if fwidth is None else fwidth
    core, top_cladding, bottom_cladding = create_pbs_mediums(mp, cfg, fcen)
    parity = get_pbs_mode_parity(mp, cfg, pol)

    flattened_weights = flatten_design_weights(struct, cfg)

    design_variables = mp.MaterialGrid(
        make_design_grid(mp, cfg),
        top_cladding,
        core,
        grid_type="U_MEAN",
    )
    design_variables.update_weights(flattened_weights)

    design_region = mpa.DesignRegion(
        design_variables,
        volume=get_design_volume(mp, cfg),
    )

    source = [
        mp.EigenModeSource(
            mp.GaussianSource(
                frequency=fcen,
                fwidth=fwidth,
            ),
            eig_parity=parity,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=_vec3(mp, cfg.kpoint_x, 0, 0),
            size=get_source_size(mp, cfg),
            center=_vec3(mp, cfg.source_x, 0, 0),
        )
    ]

    if cfg.simulation_dim == 3:
        boundary_layers = [
            mp.PML(cfg.pml_x or cfg.pml_thickness, direction=mp.X),
            mp.PML(cfg.pml_y or cfg.pml_thickness, direction=mp.Y),
            mp.PML(cfg.pml_z or cfg.pml_thickness, direction=mp.Z),
        ]
    else:
        boundary_layers = [
            mp.PML(cfg.pml_x or cfg.pml_thickness, direction=mp.X),
            mp.PML(cfg.pml_y or cfg.pml_thickness, direction=mp.Y),
        ]

    sim = mp.Simulation(
        cell_size=get_cell_size(mp, cfg),
        boundary_layers=boundary_layers,
        geometry=build_stack_geometry(mp, cfg, core, top_cladding, bottom_cladding)
        + build_pbs_geometry(mp, design_variables, cfg, core),
        sources=source,
        eps_averaging=True,
        subpixel_tol=1e-4,
        resolution=cfg.resolution,
        default_material=top_cladding,
    )

    return {
        "cfg": cfg,
        "sim": sim,
        "core": core,
        "cladding": top_cladding,
        "parity": parity,
        "design_variables": design_variables,
        "design_region": design_region,
        "flattened_weights": flattened_weights.copy(),
    }


def build_stack_geometry(mp, cfg, core, top_cladding, bottom_cladding):
    if cfg.simulation_dim != 3:
        return []
    geometry = []
    if cfg.bottom_cladding_thickness_um > 0:
        geometry.append(
            mp.Block(
                center=_vec3(
                    mp,
                    z=-0.5 * cfg.cell_size_z + 0.5 * cfg.bottom_cladding_thickness_um,
                ),
                material=bottom_cladding,
                size=_vec3(mp, cfg.cell_size_x, cfg.cell_size_y, cfg.bottom_cladding_thickness_um),
            )
        )
    if cfg.top_cladding_thickness_um > 0:
        geometry.append(
            mp.Block(
                center=_vec3(
                    mp,
                    z=0.5 * cfg.cell_size_z - 0.5 * cfg.top_cladding_thickness_um,
                ),
                material=top_cladding,
                size=_vec3(mp, cfg.cell_size_x, cfg.cell_size_y, cfg.top_cladding_thickness_um),
            )
        )
    return geometry


def build_pbs_ports(mp, mpa, sim, cfg, parity):
    layout = get_transverse_port_layout(cfg)
    port_source = mpa.EigenmodeCoefficient(
        sim,
        get_monitor_volume(
            mp,
            cfg,
            cfg.monitor_x_in,
            layout["input_center_y"],
            layout["input_monitor_span_y"],
        ),
        mode=1,
        eig_parity=parity,
    )
    port_top = mpa.EigenmodeCoefficient(
        sim,
        get_monitor_volume(
            mp,
            cfg,
            cfg.monitor_x_out,
            layout["output_center_y"],
            layout["output_monitor_span_y"],
        ),
        mode=1,
        eig_parity=parity,
    )
    port_bottom = mpa.EigenmodeCoefficient(
        sim,
        get_monitor_volume(
            mp,
            cfg,
            cfg.monitor_x_out,
            -layout["output_center_y"],
            layout["output_monitor_span_y"],
        ),
        mode=1,
        eig_parity=parity,
    )
    return port_source, port_top, port_bottom
