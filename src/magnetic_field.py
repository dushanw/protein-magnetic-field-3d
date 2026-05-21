"""
Magnetic dipole field around a set of point dipoles (the Fe atoms of hemoglobin).

Physical model
--------------
Each high-spin Fe(II) in deoxyhemoglobin has a magnetic moment of roughly
mu_Fe = 5.4 mu_B (Pauling 1936; Cerdonio et al. 1977) where mu_B is the Bohr
magneton. In an external field B0 (e.g. a 3-Tesla MRI scanner) the spins
align preferentially along B0 and produce a dipolar perturbation field
DeltaB(r) at every point r in space:

    DeltaB(r) = (mu0 / 4*pi) * sum_i [ 3 (m_i . r_hat_i) r_hat_i - m_i ] / |r_i|^3

where r_i = r - r0_i is the vector from each Fe at r0_i to the field point r.

For visualization we report DeltaB in nano-Tesla (nT) at points on a regular
3D grid surrounding the protein. We use Angstroms internally and convert to
meters only inside the prefactor; this keeps coordinates in protein-friendly
units while reporting fields in SI.

Notes
-----
* The model is a point-dipole superposition - it is the correct *far-field*
  for paramagnetic Fe in a uniform B0 and is what is used in standard BOLD
  contrast modelling. Inside the heme plane it ceases to be quantitative.
* When the user switches to the diamagnetic ("oxy") state the per-Fe moment
  drops to ~0; we expose this as a scalar `moment_scale` knob in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

MU_0 = 4.0e-7 * np.pi              # H/m, exact
MU_B = 9.2740100783e-24            # J/T, Bohr magneton
ANG_TO_M = 1.0e-10                 # 1 Angstrom in meters

# Effective magnetic moment per high-spin Fe(II) in deoxyhemoglobin (Bohr magnetons).
DEOXY_FE_MOMENT_BOHR = 5.4
# Effective moment per low-spin Fe in oxyhemoglobin (essentially diamagnetic).
OXY_FE_MOMENT_BOHR = 0.0


@dataclass
class FieldResult:
    """Outputs of a dipole-field evaluation on a regular 3D grid."""

    x: np.ndarray              # 1D grid, Angstroms
    y: np.ndarray              # 1D grid, Angstroms
    z: np.ndarray              # 1D grid, Angstroms
    Bx: np.ndarray             # 3D, shape (nx, ny, nz), Tesla
    By: np.ndarray
    Bz: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        """|DeltaB| in Tesla."""
        return np.sqrt(self.Bx ** 2 + self.By ** 2 + self.Bz ** 2)

    @property
    def magnitude_nT(self) -> np.ndarray:
        """|DeltaB| in nano-Tesla (convenient for hemoglobin-scale fields)."""
        return self.magnitude * 1.0e9

    @property
    def spacing(self) -> Tuple[float, float, float]:
        return (
            float(self.x[1] - self.x[0]) if self.x.size > 1 else 1.0,
            float(self.y[1] - self.y[0]) if self.y.size > 1 else 1.0,
            float(self.z[1] - self.z[0]) if self.z.size > 1 else 1.0,
        )

    @property
    def origin(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.y[0]), float(self.z[0])


def make_grid(
    center: np.ndarray,
    half_extent: float = 30.0,
    n: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an isotropic regular grid centered on `center`, in Angstroms.

    Parameters
    ----------
    center : (3,) array, Angstroms
    half_extent : half-side length in Angstroms (grid spans center +/- this)
    n : number of samples along each axis
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    x = np.linspace(cx - half_extent, cx + half_extent, n)
    y = np.linspace(cy - half_extent, cy + half_extent, n)
    z = np.linspace(cz - half_extent, cz + half_extent, n)
    return x, y, z


def dipole_field(
    dipole_positions: np.ndarray,  # (N, 3), Angstroms
    dipole_moments: np.ndarray,    # (N, 3), J/T (SI)
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    softening_A: float = 0.5,
) -> FieldResult:
    """
    Sum of point-dipole magnetic fields on a regular 3D grid.

    Parameters
    ----------
    dipole_positions : (N, 3) array in Angstroms.
    dipole_moments   : (N, 3) array of magnetic moment vectors in J/T.
    grid_x, grid_y, grid_z : 1D coordinate arrays in Angstroms.
    softening_A : Plummer-style softening length in Angstroms - prevents the
                  1/r^3 singularity right at the Fe atom from blowing up.

    Returns
    -------
    FieldResult with B-field components in Tesla on the (nx, ny, nz) grid.
    """
    pos = np.asarray(dipole_positions, dtype=float).reshape(-1, 3)
    mom = np.asarray(dipole_moments, dtype=float).reshape(-1, 3)
    if pos.shape != mom.shape:
        raise ValueError("dipole_positions and dipole_moments must have the same shape")

    nx, ny, nz = grid_x.size, grid_y.size, grid_z.size
    Bx = np.zeros((nx, ny, nz), dtype=float)
    By = np.zeros((nx, ny, nz), dtype=float)
    Bz = np.zeros((nx, ny, nz), dtype=float)

    # Build the field-point coordinates with broadcasting.
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")

    prefactor = MU_0 / (4.0 * np.pi)
    soft2 = (softening_A * ANG_TO_M) ** 2

    for (px, py, pz), (mx, my, mz) in zip(pos, mom):
        rx = (X - px) * ANG_TO_M
        ry = (Y - py) * ANG_TO_M
        rz = (Z - pz) * ANG_TO_M
        r2 = rx * rx + ry * ry + rz * rz + soft2
        r = np.sqrt(r2)
        inv_r3 = 1.0 / (r2 * r)            # 1 / r^3
        inv_r5 = inv_r3 / r2               # 1 / r^5
        m_dot_r = mx * rx + my * ry + mz * rz
        Bx += prefactor * (3.0 * m_dot_r * rx * inv_r5 - mx * inv_r3)
        By += prefactor * (3.0 * m_dot_r * ry * inv_r5 - my * inv_r3)
        Bz += prefactor * (3.0 * m_dot_r * rz * inv_r5 - mz * inv_r3)

    return FieldResult(x=grid_x, y=grid_y, z=grid_z, Bx=Bx, By=By, Bz=Bz)


def induced_moments(
    n_dipoles: int,
    b0_direction: np.ndarray,
    moment_per_fe_bohr: float = DEOXY_FE_MOMENT_BOHR,
    moment_scale: float = 1.0,
) -> np.ndarray:
    """
    Build dipole moment vectors, all aligned with the external field B0.

    Parameters
    ----------
    n_dipoles : number of Fe centers.
    b0_direction : (3,) array; only the direction matters, it's renormalized.
    moment_per_fe_bohr : magnitude of each Fe moment in Bohr magnetons.
    moment_scale : multiplicative knob (1.0 = deoxy, 0.0 = fully oxy).

    Returns
    -------
    (n_dipoles, 3) array of moments in J/T.
    """
    d = np.asarray(b0_direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        raise ValueError("b0_direction must be non-zero")
    d_hat = d / norm
    m_vec = moment_per_fe_bohr * MU_B * moment_scale * d_hat
    return np.tile(m_vec, (n_dipoles, 1))


if __name__ == "__main__":
    # Sanity check: a single dipole along z. Field at an equatorial point
    # should equal -mu0 * m / (4 pi r^3); at an axial point it should be +2x.
    pos = np.array([[0.0, 0.0, 0.0]])
    m_val = 5.4 * MU_B
    mom = np.array([[0.0, 0.0, m_val]])
    # Two field points placed off the origin so we never divide by zero:
    # P1 = (10, 0, 0)  -> equator,    P2 = (10, 0, 10).
    x = np.array([10.0])
    y = np.array([0.0])
    z = np.array([0.0, 10.0])
    res = dipole_field(pos, mom, x, y, z, softening_A=0.0)
    r_m = 10.0 * ANG_TO_M
    expected_eq = -MU_0 * m_val / (4.0 * np.pi * r_m ** 3)
    r2_m = 10.0 * np.sqrt(2.0) * ANG_TO_M
    expected_at_45 = (
        MU_0 / (4.0 * np.pi)
        * (3.0 * m_val * (10.0 * ANG_TO_M) ** 2 / r2_m ** 5 - m_val / r2_m ** 3)
    )
    print(f"Equatorial Bz @(10,0,0) (computed): {res.Bz[0, 0, 0]:.3e} T")
    print(f"Equatorial Bz @(10,0,0) (expected): {expected_eq:.3e} T")
    print(f"45deg     Bz @(10,0,10)(computed): {res.Bz[0, 0, 1]:.3e} T")
    print(f"45deg     Bz @(10,0,10)(expected): {expected_at_45:.3e} T")
