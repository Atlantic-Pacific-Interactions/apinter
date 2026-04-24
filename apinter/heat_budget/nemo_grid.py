"""NEMO ORCA025 C-grid operations for the mixed-layer heat budget.

Loads mesh_mask.nc and provides finite-difference operators on the native
Arakawa C-grid (T, U, V staggering).

Grid layout (Arakawa C)::

        +------V[j,i]------+
        |                   |
      U[j,i-1]   T[j,i]  U[j,i]
        |                   |
        +----V[j-1,i]------+

Reference: Madec, G. & the NEMO System Team (2019). NEMO ocean engine. Chapter 4.
"""
from pathlib import Path

import xarray as xr

from apinter.config import ORAS5_MESH_PATH


class NemoGrid:
    """Container for NEMO ORCA025 grid metrics, masks, and C-grid operators.

    Parameters
    ----------
    mesh_path : str or Path, optional
        Path to ``mesh_mask.nc``. Defaults to ``apinter.config.ORAS5_MESH_PATH``.
    """

    def __init__(self, mesh_path=None):
        if mesh_path is None:
            mesh_path = ORAS5_MESH_PATH
        self._load(Path(mesh_path))

    def _load(self, mesh_path):
        """Load grid metrics from mesh_mask.nc."""
        ds = xr.open_dataset(mesh_path)

        # --- Horizontal grid metrics (squeeze singleton 't' dim) ---
        self.e1t = ds['e1t'].squeeze(drop=True)   # T-cell x-width [m]
        self.e2t = ds['e2t'].squeeze(drop=True)   # T-cell y-width [m]
        self.e1u = ds['e1u'].squeeze(drop=True)   # U-cell x-width [m]
        self.e2u = ds['e2u'].squeeze(drop=True)   # U-cell y-width [m]
        self.e1v = ds['e1v'].squeeze(drop=True)   # V-cell x-width [m]
        self.e2v = ds['e2v'].squeeze(drop=True)   # V-cell y-width [m]

        # --- Vertical grid metrics ---
        self.e3t_0 = ds['e3t_0'].squeeze(drop=True)  # layer thickness [m] (z,)
        self.e3w_0 = ds['e3w_0'].squeeze(drop=True)  # w-level spacing [m] (z,)
        self.gdept_0 = ds['gdept_0'].squeeze(drop=True)  # T-point depths [m] (z,)
        self.gdepw_0 = ds['gdepw_0'].squeeze(drop=True)  # W-point depths [m] (z,)

        # --- Masks (3D: z, y, x) ---
        self.tmask = ds['tmask'].squeeze(drop=True)  # 1=ocean, 0=land
        self.umask = ds['umask'].squeeze(drop=True)
        self.vmask = ds['vmask'].squeeze(drop=True)

        # --- 2D surface masks ---
        self.tmaskutil = ds['tmaskutil'].squeeze(drop=True)
        self.umaskutil = ds['umaskutil'].squeeze(drop=True)
        self.vmaskutil = ds['vmaskutil'].squeeze(drop=True)

        # --- Coordinates (2D: y, x) ---
        self.glamt = ds['glamt'].squeeze(drop=True)  # T-point longitude
        self.gphit = ds['gphit'].squeeze(drop=True)  # T-point latitude
        self.glamu = ds['glamu'].squeeze(drop=True)
        self.gphiu = ds['gphiu'].squeeze(drop=True)
        self.glamv = ds['glamv'].squeeze(drop=True)
        self.gphiv = ds['gphiv'].squeeze(drop=True)

        # --- Grid shape ---
        self.ny = ds.sizes['y']
        self.nx = ds.sizes['x']
        self.nz = ds.sizes['z']

        # --- Coriolis parameter ---
        self.ff = ds['ff'].squeeze(drop=True)

        ds.close()

    # ------------------------------------------------------------------
    # Gradient operators (scalar field → flux-point gradient)
    # ------------------------------------------------------------------

    def grad_i(self, field):
        """Zonal gradient of a T-point field, result at U-point.

        grad_i(T)[j,i] = (T[j,i+1] - T[j,i]) / e1u[j,i]

        Parameters
        ----------
        field : xr.DataArray (..., y, x)
            Scalar field on T-points.

        Returns
        -------
        xr.DataArray (..., y, x)
            Gradient at U-points.  The last x-column is invalid (boundary).
        """
        dfield = field.roll(x=-1, roll_coords=False) - field
        return dfield / self.e1u

    def grad_j(self, field):
        """Meridional gradient of a T-point field, result at V-point.

        grad_j(T)[j,i] = (T[j+1,i] - T[j,i]) / e2v[j,i]

        Parameters
        ----------
        field : xr.DataArray (..., y, x)

        Returns
        -------
        xr.DataArray (..., y, x)
            Gradient at V-points.  The last y-row is invalid (boundary).
        """
        dfield = field.roll(y=-1, roll_coords=False) - field
        return dfield / self.e2v

    # ------------------------------------------------------------------
    # Advection at T-point (averaging adjacent U/V fluxes)
    # ------------------------------------------------------------------

    def u_dot_grad_i(self, u, scalar):
        """Zonal advection u·∂T/∂x at T-point.

        Average of the eastern and western flux divergences:
        u·dT/dx|_T[j,i] = 0.5 * (
            u[j,i]   * (T[j,i+1] - T[j,i])   / e1u[j,i]
          + u[j,i-1] * (T[j,i]   - T[j,i-1]) / e1u[j,i-1]
        )

        Parameters
        ----------
        u : xr.DataArray (..., y, x)
            Zonal velocity at U-points.
        scalar : xr.DataArray (..., y, x)
            Scalar field at T-points.

        Returns
        -------
        xr.DataArray (..., y, x)
            u·∂scalar/∂x at T-points.
        """
        # Eastern contribution: u[j,i] * (T[j,i+1] - T[j,i]) / e1u[j,i]
        flux_east = u * self.grad_i(scalar)

        # Western contribution: u[j,i-1] * (T[j,i] - T[j,i-1]) / e1u[j,i-1]
        flux_west = flux_east.roll(x=1, roll_coords=False)

        return 0.5 * (flux_east + flux_west)

    def v_dot_grad_j(self, v, scalar):
        """Meridional advection v·∂T/∂y at T-point.

        v·dT/dy|_T[j,i] = 0.5 * (
            v[j,i]   * (T[j+1,i] - T[j,i])   / e2v[j,i]
          + v[j-1,i] * (T[j,i]   - T[j-1,i]) / e2v[j-1,i]
        )

        Parameters
        ----------
        v : xr.DataArray (..., y, x)
            Meridional velocity at V-points.
        scalar : xr.DataArray (..., y, x)
            Scalar field at T-points.

        Returns
        -------
        xr.DataArray (..., y, x)
            v·∂scalar/∂y at T-points.
        """
        flux_north = v * self.grad_j(scalar)
        flux_south = flux_north.roll(y=1, roll_coords=False)
        return 0.5 * (flux_north + flux_south)

    # ------------------------------------------------------------------
    # Divergence at T-point (for continuity equation)
    # ------------------------------------------------------------------

    def div_h(self, u, v):
        """Horizontal divergence at T-point.

        div = (1 / e1t·e2t) * [
            u[j,i]·e2u[j,i] - u[j,i-1]·e2u[j,i-1]
          + v[j,i]·e1v[j,i] - v[j-1,i]·e1v[j-1,i]
        ]

        Parameters
        ----------
        u : xr.DataArray (..., y, x) at U-points.
        v : xr.DataArray (..., y, x) at V-points.

        Returns
        -------
        xr.DataArray (..., y, x) at T-points.
        """
        # Zonal flux difference
        u_flux = u * self.e2u
        du = u_flux - u_flux.roll(x=1, roll_coords=False)

        # Meridional flux difference
        v_flux = v * self.e1v
        dv = v_flux - v_flux.roll(y=1, roll_coords=False)

        area = self.e1t * self.e2t
        return (du + dv) / area

    # ------------------------------------------------------------------
    # Gradient of a T-point 2D field (for MLD gradient in entrainment)
    # ------------------------------------------------------------------

    def grad_i_at_T(self, field):
        """Zonal gradient of a T-point field, result back at T-point.

        Centered difference using two adjacent U-point gradients.

        Parameters
        ----------
        field : xr.DataArray (..., y, x) at T-points.

        Returns
        -------
        xr.DataArray (..., y, x) at T-points.
        """
        grad_u = self.grad_i(field)  # at U-points
        # Average U[j,i] and U[j,i-1] back to T[j,i]
        return 0.5 * (grad_u + grad_u.roll(x=1, roll_coords=False))

    def grad_j_at_T(self, field):
        """Meridional gradient of a T-point field, result back at T-point.

        Parameters
        ----------
        field : xr.DataArray (..., y, x) at T-points.

        Returns
        -------
        xr.DataArray (..., y, x) at T-points.
        """
        grad_v = self.grad_j(field)  # at V-points
        return 0.5 * (grad_v + grad_v.roll(y=1, roll_coords=False))

    # ------------------------------------------------------------------
    # Velocity interpolation (U/V → T-point)
    # ------------------------------------------------------------------

    def u_to_T(self, u):
        """Interpolate U-point field to T-point (simple average).

        u_T[j,i] = 0.5 * (u[j,i] + u[j,i-1])
        """
        return 0.5 * (u + u.roll(x=1, roll_coords=False))

    def v_to_T(self, v):
        """Interpolate V-point field to T-point (simple average).

        v_T[j,i] = 0.5 * (v[j,i] + v[j-1,i])
        """
        return 0.5 * (v + v.roll(y=1, roll_coords=False))
