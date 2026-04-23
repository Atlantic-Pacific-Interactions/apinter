"""Omega lead-lag regression profile plots (longitude × pressure).

Ported from ``Paper_1/src/omega_reg_plotting.py``. The original hardcoded
Paper_1 figures directory is removed; the caller passes ``output_dir``
(or ``None`` to skip saving). The region labels (Indian / WP / CEP /
Atlantic) and vertical separators are retained — they're diagnostic of
the Paper_1 omega-regression analysis.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _convert_to_hPa(tick_value: float, pos: int) -> int:
    """Axis-tick formatter: Pa -> hPa."""
    return int(tick_value / 100)


def _lag_title(lag_months: int) -> str:
    """Turn a lag in months into a human-friendly lead/lag years string."""
    lead_lag_months = -lag_months
    yrs = lead_lag_months / 12
    if yrs > 0:
        return f'Lead {int(yrs) if yrs == int(yrs) else f"{yrs:.1f}"} years'
    if yrs < 0:
        a = abs(yrs)
        return f'Lag {int(a) if a == int(a) else f"{a:.1f}"} years'
    return 'Concurrent'


def plot_omega_lead_lag_profile(
    data_dict: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str,
    index_name: str = 'tpdv',
    var: str = 'regression',
    vmin: float = -0.003,
    vmax: float = 0.003,
    interval: float = 0.0005,
    thinning_factor_lon: int = 3,
    figsize: Tuple[int, int] = (30, 18),
    savepath: Optional[str] = None,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """Lead-lag omega profiles for one (model, index) across all lags.

    Parameters
    ----------
    data_dict : dict
        Nested: ``data_dict[index_name][lag][var]`` → xr.DataArray with dims
        (lev, lon). An optional ``'significant'`` key at each lag triggers
        stippling.
    model_name : str
    index_name : {'tpdv', 'tamv'} or similar — must be a key of ``data_dict``.
    var : str
        Which variable of the lag dict to plot (typically ``'regression'``
        or ``'correlation'``).
    vmin, vmax, interval : float
        Contour range and spacing.
    thinning_factor_lon : int
        Subsampling factor for significance stippling.
    figsize : (w, h)
    savepath : str, optional
        Full path (including filename) to save the figure. None = no save.

    Returns
    -------
    (fig, axes) or (None, None) if the index has no data.
    """
    if index_name not in data_dict:
        logger.warning(f"Index {index_name} not found in data for model {model_name}")
        return None, None

    lag_times = sorted(data_dict[index_name].keys())
    n_lags = len(lag_times)
    if n_lags == 0:
        logger.warning(f"No lag data for {index_name} / {model_name}")
        return None, None

    n_cols = min(7, n_lags)
    n_rows = (n_lags + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=200)
    axs = np.atleast_1d(axs).flatten()
    cf = None

    for idx, lag in enumerate(lag_times):
        ax = axs[idx]
        ds = data_dict[index_name][lag][var]

        # Normalize pressure-coord name to 'lev' and choose y-unit handling
        y_is_pa = False
        if 'plev' in ds.dims:
            ds = ds.rename({'plev': 'lev'})
            y_is_pa = True
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_convert_to_hPa))
            y_position = 950 * 100
        elif 'level' in ds.dims:
            ds = ds.rename({'level': 'lev'})
            y_position = 950
        else:
            y_position = 950

        if 'longitude' in ds.dims:
            ds = ds.rename({'longitude': 'lon'})

        ax.text(55, y_position, 'Indian', fontsize=8, fontweight='bold')
        ax.text(130, y_position, 'WP', fontsize=8, fontweight='bold')
        ax.text(220, y_position, 'CEP', fontsize=8, fontweight='bold')
        ax.text(300, y_position, 'Atlantic', fontsize=8, fontweight='bold')

        levels = np.arange(vmin, vmax + interval, interval)
        cf = ax.contourf(ds['lon'], ds['lev'], ds.values,
                         cmap='RdBu_r', levels=levels, extend='both')

        for x in (30, 110, 180, 280):
            ax.axvline(x, color='k', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Longitude', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel('Pressure (hPa)', fontsize=10)

        ax.set_ylim(100, 1000)
        ax.set_yscale('log')
        pressure_levels = [1000, 850, 700, 600, 500, 400, 300, 200]
        ax.set_yticks(pressure_levels)
        ax.set_yticklabels([str(p) for p in pressure_levels])
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=8, width=1)
        ax.tick_params(axis='both', which='minor', labelsize=8, width=1)
        ax.set_xlim(0, 360)

        for axis in ('top', 'bottom', 'left', 'right'):
            ax.spines[axis].set_linewidth(1)

        ax.set_title(_lag_title(lag), loc='left', fontweight='bold')
        ax.set_title(model_name, loc='right', fontweight='bold', fontsize=8)

        # Optional significance stippling
        if 'significant' in data_dict[index_name][lag]:
            try:
                sig = data_dict[index_name][lag]['significant']
                if 'plev' in sig.dims:
                    sig = sig.rename({'plev': 'lev'})
                elif 'level' in sig.dims:
                    sig = sig.rename({'level': 'lev'})
                if 'longitude' in sig.dims:
                    sig = sig.rename({'longitude': 'lon'})
                lons, levs = np.meshgrid(sig['lon'], sig['lev'])
                mask = sig.values == True
                t = thinning_factor_lon
                ax.scatter(lons[:, ::t][mask[:, ::t]],
                           levs[:, ::t][mask[:, ::t]],
                           color='k', marker='.', s=5, alpha=0.6)
            except Exception as e:
                logger.warning(f"stippling failed for lag {lag}: {e}")

    for idx in range(n_lags, len(axs)):
        axs[idx].set_visible(False)

    if cf is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(cf, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        if var == 'regression':
            cbar.set_label('Regression Coefficient (Pa/s)',
                           fontsize=12, fontweight='bold')
        elif var == 'correlation':
            cbar.set_label('Correlation Coefficient',
                           fontsize=12, fontweight='bold')
        else:
            cbar.set_label(var.capitalize(), fontsize=12, fontweight='bold')

    fig.suptitle(f'{index_name.upper()}-Omega Lead-Lag {var.capitalize()} — {model_name}',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {savepath}")

    return fig, axs


def create_omega_regression_plots(
    model_results: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str,
    output_dir: Optional[str] = None,
    vmin: float = -0.003,
    vmax: float = 0.003,
    interval: float = 0.0005,
    thinning_factor_lon: int = 1,
    var: str = 'regression',
) -> None:
    """Create omega lead-lag plots for every index in ``model_results``.

    Parameters
    ----------
    model_results : dict
        Nested ``{index_name: {lag: {'regression': DataArray, ...}}}``.
    model_name : str
    output_dir : str, optional
        Directory where per-index PNGs land. None disables saving.
    vmin, vmax, interval : float
    thinning_factor_lon : int
    var : str
        Variable to render (``'regression'`` by default).
    """
    for index_name in model_results:
        savepath = None
        if output_dir is not None:
            savepath = (Path(output_dir) /
                        f"{model_name}_{index_name}_omega_lead_lag_{var}.png")
        plot_omega_lead_lag_profile(
            data_dict=model_results,
            model_name=model_name,
            index_name=index_name,
            var=var,
            vmin=vmin,
            vmax=vmax,
            interval=interval,
            thinning_factor_lon=thinning_factor_lon,
            savepath=str(savepath) if savepath else None,
        )
