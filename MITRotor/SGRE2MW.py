# SGRE2MW.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d

from .RotorDefinition import RotorDefinition


def _linear_extrap_from_uniform_grid(xgrid: np.ndarray, ygrid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Vectorized 1D interpolation on a uniform grid with linear extrapolation.
    xgrid must be strictly increasing and uniform spacing.
    """
    x = np.asarray(x)
    y = np.interp(x, xgrid, ygrid)  # clamps outside; we'll fix ends

    # Left extrapolation
    left = x < xgrid[0]
    if np.any(left):
        slope = (ygrid[1] - ygrid[0]) / (xgrid[1] - xgrid[0])
        y[left] = ygrid[0] + slope * (x[left] - xgrid[0])

    # Right extrapolation
    right = x > xgrid[-1]
    if np.any(right):
        slope = (ygrid[-1] - ygrid[-2]) / (xgrid[-1] - xgrid[-2])
        y[right] = ygrid[-1] + slope * (x[right] - xgrid[-1])

    return y


@dataclass
class SGRE21MWAirfoilEvaluator:
    """
    MATLAB-compatible airfoil evaluation for the SGRE 2.1MW turbine:

    - Each radial station has two bounding airfoils (ID1, ID2).
    - For a given alpha, compute (CL1, CD1) and (CL2, CD2) from those two airfoils.
    - Linearly blend CL/CD by thickness between (T1, T2) per element:
        CL = CL1*(T2 - t)/(T2 - T1) + CL2*(t - T1)/(T2 - T1)
      same for CD.

    Notes:
    - IDs in the exported file are 0-based (as in AeroDyn.AeroID* before MATLAB +1).
    - Alpha grid used for evaluation is uniform: -180..180 deg.
    """
    mu_grid: np.ndarray              # 1D: r / R, increasing
    thickness_grid: np.ndarray       # 1D: local thickness at mu_grid
    id1_grid: np.ndarray             # 1D: 0-based airfoil id at mu_grid
    id2_grid: np.ndarray             # 1D: 0-based airfoil id at mu_grid
    how_thick: np.ndarray            # 1D: thickness of each airfoil type (len nFoils)

    alpha_grid_deg: np.ndarray       # 1D uniform: -180..180
    cl_table: np.ndarray             # 2D: [nFoils, nAlpha]
    cd_table: np.ndarray             # 2D: [nFoils, nAlpha]

    def __call__(self, mu: np.ndarray, aoa_rad: np.ndarray):
        mu = np.asarray(mu)
        aoa_rad = np.asarray(aoa_rad)
        aoa_deg = np.rad2deg(aoa_rad)

        mu_flat = mu.ravel()
        aoa_flat = aoa_deg.ravel()

        # Interpolate thickness and ids along radius (match MATLAB interp1(...,'linear','extrap'))
        thickness = np.interp(mu_flat, self.mu_grid, self.thickness_grid)

        id1_f = np.interp(mu_flat, self.mu_grid, self.id1_grid.astype(float))
        id2_f = np.interp(mu_flat, self.mu_grid, self.id2_grid.astype(float)) 

        # MATLAB uses ids that are effectively piecewise-constant; enforce integer ids safely
        id1 = np.clip(np.rint(id1_f).astype(int), 0, self.cl_table.shape[0] - 1)
        id2 = np.clip(np.rint(id2_f).astype(int), 0, self.cl_table.shape[0] - 1)

        # Airfoil thickness values for the two bounding foils
        T1 = self.how_thick[id1]
        T2 = self.how_thick[id2]

        denom = (T2 - T1)
        # Avoid divide-by-zero (if T1==T2, no blend needed)
        safe = np.abs(denom) > 1e-12
        w2 = np.zeros_like(thickness)
        w2[safe] = (thickness[safe] - T1[safe]) / denom[safe]
        w2[~safe] = 0.0
        w1 = 1.0 - w2

        # Lookup CL/CD from uniform alpha grid with linear extrapolation
        cl1 = np.empty_like(aoa_flat, dtype=float)
        cl2 = np.empty_like(aoa_flat, dtype=float)
        cd1 = np.empty_like(aoa_flat, dtype=float)
        cd2 = np.empty_like(aoa_flat, dtype=float)

        # Evaluate per unique id to avoid per-element Python loops
        for ids, out_cl, out_cd in [(id1, cl1, cd1), (id2, cl2, cd2)]:
            for uid in np.unique(ids):
                mask = (ids == uid)
                out_cl[mask] = _linear_extrap_from_uniform_grid(
                    self.alpha_grid_deg, self.cl_table[uid], aoa_flat[mask]
                )
                out_cd[mask] = _linear_extrap_from_uniform_grid(
                    self.alpha_grid_deg, self.cd_table[uid], aoa_flat[mask]
                )

        cl = w1 * cl1 + w2 * cl2
        cd = w1 * cd1 + w2 * cd2

        return cl.reshape(mu.shape), cd.reshape(mu.shape)


def load_sgre_2mw_from_npz(npz_path: str) -> RotorDefinition:
    """
    Create a RotorDefinition from the exported SGRE_2MW_python.npz.

    This returns a standard RotorDefinition instance, so your solver remains unchanged.
    """
    d = np.load(npz_path, allow_pickle=True)

    R = float(d["R"])
    hub_radius = float(d["hub_radius"])
    hub_height = float(d["hub_height"])
    N_blades = int(d["B"])
    tsr_target = float(d["tsr_target"])

    r = np.asarray(d["r"], dtype=float).reshape(-1)
    chord = np.asarray(d["chord"], dtype=float).reshape(-1)
    twist_deg = np.asarray(d["twist_deg"], dtype=float).reshape(-1)
    thickness = np.asarray(d["thickness"], dtype=float).reshape(-1)
    aero_id1 = np.asarray(d["aero_id1"], dtype=float).reshape(-1)
    aero_id2 = np.asarray(d["aero_id2"], dtype=float).reshape(-1)

    mu_grid = r / R

    # Functions expected by RotorDefinition are in terms of mu = r/R
    chord_func = interp1d(mu_grid, chord, kind="linear", fill_value="extrapolate")
    twist_rad = np.deg2rad(twist_deg)
    twist_func = interp1d(mu_grid, twist_rad, kind="linear", fill_value="extrapolate")

    # Keep "old" funcs for compatibility with your existing RotorDefinition signature
    chord_func_old = chord_func
    twist_func_old = twist_func

    solidity_func = lambda mu: np.minimum(
        N_blades * chord_func(mu) / (2 * np.pi * np.maximum(mu, 1e-4) * R),
        1.0,
    )

    how_thick = np.asarray(d["HowThick"], dtype=float).reshape(-1)

    # Airfoil polars (cells) -> uniform alpha grid + tables
    alpha_cell = d["alpha_cell"]
    cl_cell = d["cl_cell"]
    cd_cell = d["cd_cell"]

    nfoils = len(alpha_cell)
    alpha_grid_deg = np.arange(-180.0, 181.0, 1.0)

    cl_table = np.zeros((nfoils, alpha_grid_deg.size), dtype=float)
    cd_table = np.zeros((nfoils, alpha_grid_deg.size), dtype=float)

    for i in range(nfoils):
        alpha_i = np.asarray(alpha_cell[i], dtype=float).reshape(-1)
        cl_i = np.asarray(cl_cell[i], dtype=float).reshape(-1)
        cd_i = np.asarray(cd_cell[i], dtype=float).reshape(-1)

        # MATLAB ComputePolars uses interp1(...,'linear') without explicit extrapolate here;
        # but later griddedInterpolant may extrapolate. We match with linear extrapolation.
        cl_table[i] = interp1d(alpha_i, cl_i, kind="linear", fill_value="extrapolate")(alpha_grid_deg)
        cd_table[i] = interp1d(alpha_i, cd_i, kind="linear", fill_value="extrapolate")(alpha_grid_deg)

    airfoil_func = SGRE21MWAirfoilEvaluator(
        mu_grid=mu_grid,
        thickness_grid=thickness,
        id1_grid=aero_id1.astype(int),
        id2_grid=aero_id2.astype(int),
        how_thick=how_thick,
        alpha_grid_deg=alpha_grid_deg,
        cl_table=cl_table,
        cd_table=cd_table,
    )

    return RotorDefinition(
        twist_func=twist_func,
        # twist_func_old=twist_func_old,
        chord_func=chord_func,
        # chord_func_old=chord_func_old,
        solidity_func=solidity_func,
        airfoil_func=airfoil_func,
        N_blades=N_blades,
        R=R,
        P_rated=np.nan,              # you can export rated power too if you want
        rotorspeed_max=np.nan,       # optional
        hub_height=hub_height,
        tsr_target=tsr_target,
        hub_radius=hub_radius,
        name="SGRE 2.1 MW",
    )