from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PBSPlatformConfig:
    name: str
    simulation_dim: int
    cladding_index: float
    core_index: Optional[float] = None
    core_epsilon_diag: Optional[Tuple[float, float, float]] = None
    top_cladding_index: Optional[float] = None
    bottom_cladding_index: Optional[float] = None
    resolution: int = 21
    cell_size_x: float = 10.0
    cell_size_y: float = 10.0
    cell_size_z: float = 0.0
    pml_thickness: float = 2.0
    wavelength_um: float = 1.55
    source_width: float = 0.2
    source_x: float = -2.7
    source_span_y: float = 2.0
    source_span_z: float = 0.0
    kpoint_x: float = 1.0
    grid_nx: int = 64
    grid_ny: int = 64
    grid_nz: int = 1
    design_region_size_um: float = 3.0
    design_region_thickness_um: float = 0.0
    film_thickness_um: float = 0.0
    etch_depth_um: float = 0.0
    slab_thickness_um: float = 0.0
    top_cladding_thickness_um: float = 0.0
    bottom_cladding_thickness_um: float = 0.0
    input_wg_width: float = 1.0
    output_wg_width: float = 0.5
    waveguide_thickness_um: float = 0.0
    output_offset_y: float = 0.8
    monitor_x_in: float = -2.5
    monitor_x_out: float = 2.5
    monitor_span_y_in: float = 2.0
    monitor_span_y_out: float = 1.0
    monitor_span_z: float = 0.0
    cross_weight: float = 0.5
    dominant_field_te: str = "Ez"
    dominant_field_tm: str = "Ex"
    crystal_cut: str = "isotropic"
    meep_material_name: Optional[str] = None
    sidewall_angle_deg: float = 0.0
    loss_db_per_cm: float = 0.0
    dispersion_ref_wavelength_um: float = 1.55
    ordinary_index_ref: Optional[float] = None
    extraordinary_index_ref: Optional[float] = None
    ordinary_dispersion_slope: float = 0.0
    extraordinary_dispersion_slope: float = 0.0
    cladding_index_ref: Optional[float] = None
    cladding_dispersion_slope: float = 0.0
    objective_wavelengths_um: Optional[Tuple[float, ...]] = None
    objective_wavelength_weights: Optional[Tuple[float, ...]] = None
    optic_axis: Optional[Tuple[float, float, float]] = None
    pml_x: Optional[float] = None
    pml_y: Optional[float] = None
    pml_z: Optional[float] = None
    align_waveguides_to_pixels: bool = False


PBS_PLATFORM_CONFIGS = {
    "soi": PBSPlatformConfig(
        name="soi",
        simulation_dim=3,
        cladding_index=1.44,
        core_index=3.4,
        resolution=40,
        cell_size_z=2.22,
        source_span_z=1.2,
        input_wg_width=1.0,
        output_wg_width=0.5,
        design_region_thickness_um=0.22,
        film_thickness_um=0.22,
        top_cladding_thickness_um=1.0,
        bottom_cladding_thickness_um=1.0,
        waveguide_thickness_um=0.22,
        output_offset_y=0.8,
        monitor_span_z=1.2,
        dominant_field_te="Ez",
        dominant_field_tm="Ex",
        pml_z=0.5,
        align_waveguides_to_pixels=True,
    ),
    "tfln": PBSPlatformConfig(
        name="tfln",
        simulation_dim=3,
        cladding_index=1.44,
        meep_material_name=None,
        top_cladding_index=1.44,
        bottom_cladding_index=1.44,
        grid_nx=128,
        grid_ny=128,
        grid_nz=8,
        cell_size_x=14.0,
        cell_size_y=14.0,
        cell_size_z=8.0,
        source_span_y=2.4,
        source_span_z=1.8,
        design_region_size_um=6.0,
        design_region_thickness_um=0.35,
        film_thickness_um=0.6,
        etch_depth_um=0.35,
        slab_thickness_um=0.25,
        top_cladding_thickness_um=1.3,
        bottom_cladding_thickness_um=1.3,
        input_wg_width=1.0,
        output_wg_width=0.8,
        waveguide_thickness_um=0.35,
        output_offset_y=1.2,
        monitor_x_in=-3.5,
        monitor_x_out=3.5,
        monitor_span_y_in=2.4,
        monitor_span_y_out=1.2,
        monitor_span_z=1.8,
        dominant_field_te="Ey",
        dominant_field_tm="Ez",
        crystal_cut="z-cut",
        sidewall_angle_deg=12.0,
        loss_db_per_cm=0.2,
        dispersion_ref_wavelength_um=1.55,
        ordinary_index_ref=2.211,
        extraordinary_index_ref=2.138,
        ordinary_dispersion_slope=-0.03,
        extraordinary_dispersion_slope=-0.025,
        cladding_index_ref=1.444,
        cladding_dispersion_slope=-0.01,
        objective_wavelengths_um=(1.50, 1.55, 1.60),
        objective_wavelength_weights=(0.25, 0.5, 0.25),
        optic_axis=(0.0, 0.0, 1.0),
        pml_x=2.0,
        pml_y=2.0,
        pml_z=1.0,
    ),
}


def get_pbs_platform_config(platform: str = "soi") -> PBSPlatformConfig:
    platform_key = (platform or "soi").lower()
    if platform_key not in PBS_PLATFORM_CONFIGS:
        supported = ", ".join(sorted(PBS_PLATFORM_CONFIGS))
        raise ValueError(
            f"Unsupported PBS platform '{platform}'. Supported platforms: {supported}."
        )
    return PBS_PLATFORM_CONFIGS[platform_key]
