"""Spatial map plots: regression and trend maps.

Ported from ``./src/plot/plot_regression_map.py`` and
``./src/plot/plot_trend_map.py``. The original hardcoded output paths and
verbose-logging flags are removed; save/show decisions are left to the caller.

All inputs are expected to use ``lat`` / ``lon`` coord names (package convention).
"""
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# =============================================================================
# Regression maps
# =============================================================================

def plot_regression_map(
    ax: plt.Axes,
    data: xr.Dataset,
    variable: str = 'beta',
    title: str = '',
    vmin: float = -0.1,
    vmax: float = 0.1,
    cmap: str = 'RdBu_r',
    levels: int = 21,
    significance_alpha: float = 0.1,
    add_stippling: bool = True,
    stipple_thinning: int = 5,
    stipple_size: float = 3,
    stipple_alpha: float = 0.5,
    stipple_color: str = 'black',
    stipple_marker: str = '.',
    add_coastlines: bool = True,
    coastline_color: str = 'black',
    coastline_linewidth: float = 0.5,
    add_gridlines: bool = True,
    gridline_color: str = 'gray',
    gridline_alpha: float = 0.4,
    gridline_linestyle: str = '--',
    gridline_linewidth: float = 1.0,
    top_labels: bool = False,
    right_labels: bool = False,
    add_box: bool = False,
    box_coords: Optional[Tuple[float, float, float, float]] = None,
    box_color: str = 'black',
    box_linewidth: float = 2,
    box_linestyle: str = '-',
    box_fill: bool = False,
    box_alpha: float = 1.0,
) -> Any:
    """Regression/correlation map on a single axes.

    Parameters
    ----------
    ax : plt.Axes
        Pre-made Cartopy axes (projection supplied by the caller).
    data : xr.Dataset
        Output of :func:`apinter.stats.regression_lags` (e.g., variables
        ``beta`` and ``p_value``) or any Dataset with a spatial variable and
        an optional ``p_value`` variable for stippling.
    variable : str
        Data variable to plot (default ``beta`` — was ``slope`` in legacy).
    title : str
    vmin, vmax : float
        Colormap range.
    cmap : str
    levels : int
        Number of contour levels between vmin and vmax.
    significance_alpha : float
        p-value threshold for stippling.
    add_stippling : bool
        Whether to overlay dots at significant cells.
    stipple_thinning : int
        Subsample stippling every Nth cell in lat and lon.
    add_coastlines, add_gridlines : bool
    add_box : bool, box_coords : (lon_min, lat_min, width, height)
        Optional rectangular overlay (e.g., a TAMV / TPDV box).

    Returns
    -------
    matplotlib QuadContourSet — pass to ``fig.colorbar``.
    """
    da = data[variable]
    lons = da['lon'].values
    lats = da['lat'].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    contour_levels = np.linspace(vmin, vmax, levels)
    im = ax.contourf(
        lon2d, lat2d, da.values,
        levels=contour_levels, cmap=cmap,
        vmin=vmin, vmax=vmax, extend='both',
        transform=ccrs.PlateCarree(),
    )

    if add_stippling and 'p_value' in data:
        sig = data['p_value'].values < significance_alpha
        if sig.any():
            slon, slat = np.meshgrid(lons, lats)
            t = stipple_thinning
            sig_t = sig[::t, ::t]
            slon_t = slon[::t, ::t]
            slat_t = slat[::t, ::t]
            ax.scatter(
                slon_t[sig_t], slat_t[sig_t],
                color=stipple_color, marker=stipple_marker,
                s=stipple_size, alpha=stipple_alpha,
                transform=ccrs.PlateCarree(),
            )

    if add_coastlines:
        ax.coastlines(color=coastline_color, linewidth=coastline_linewidth)

    if add_gridlines:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=gridline_linewidth, color=gridline_color,
            alpha=gridline_alpha, linestyle=gridline_linestyle,
        )
        gl.top_labels = top_labels
        gl.right_labels = right_labels

    if add_box and box_coords is not None:
        lon_min, lat_min, width, height = box_coords
        ax.add_patch(mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            fill=box_fill, edgecolor=box_color,
            linewidth=box_linewidth, linestyle=box_linestyle,
            alpha=box_alpha, transform=ccrs.PlateCarree(),
        ))

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    return im


def plot_multiple_regression_maps(
    datasets: List[xr.Dataset],
    variable: str = 'beta',
    titles: Optional[List[str]] = None,
    suptitle: str = '',
    nrows: int = 3,
    ncols: int = 3,
    figsize: Tuple[int, int] = (18, 12),
    projection: ccrs.Projection = None,
    dpi: int = 300,
    vmin: float = -0.1,
    vmax: float = 0.1,
    cmap: str = 'RdBu_r',
    levels: int = 21,
    colorbar_label: str = 'Regression Coefficient',
    colorbar_orientation: str = 'horizontal',
    colorbar_fraction: float = 0.025,
    colorbar_pad: float = 0.1,
    colorbar_aspect: float = 50,
    savepath: Optional[str] = None,
    common_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    **plot_kwargs,
) -> plt.Figure:
    """Grid of regression maps sharing a colorbar.

    Parameters
    ----------
    datasets : list of xr.Dataset
    variable : str
    titles : list of str, optional
        Per-panel titles. Defaults to ``"Dataset 1"`` … if not provided.
    suptitle : str
    nrows, ncols : int
    figsize : (w, h)
    projection : cartopy.crs.Projection, optional
        Default Robinson(central_longitude=180) if None.
    vmin, vmax, cmap, levels : colormap options.
    colorbar_* : colorbar layout.
    savepath : str, optional
        File path for saving; no save if None.
    common_boxes : list of (lon_min, lat_min, width, height), optional
        Rectangles to draw on every panel.
    **plot_kwargs
        Forwarded to :func:`plot_regression_map`.
    """
    if projection is None:
        projection = ccrs.Robinson(central_longitude=180)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        subplot_kw={'projection': projection}, dpi=dpi,
    )
    axes = np.atleast_1d(axes).flatten()

    if titles is None:
        titles = [f'Dataset {i+1}' for i in range(len(datasets))]
    elif len(titles) < len(datasets):
        titles = list(titles) + [f'Dataset {i+1}'
                                 for i in range(len(titles), len(datasets))]

    last_im = None
    for i, (ax, ds, title) in enumerate(zip(axes, datasets, titles)):
        kw = dict(plot_kwargs)
        if common_boxes:
            # Draw each box via the single-panel function using the first as add_box,
            # then layer the rest manually so they all render.
            kw['add_box'] = True
            kw['box_coords'] = common_boxes[0]
        last_im = plot_regression_map(
            ax=ax, data=ds, variable=variable, title=title,
            vmin=vmin, vmax=vmax, cmap=cmap, levels=levels, **kw,
        )
        if common_boxes and len(common_boxes) > 1:
            for bx in common_boxes[1:]:
                lon_min, lat_min, width, height = bx
                ax.add_patch(mpatches.Rectangle(
                    (lon_min, lat_min), width, height, fill=False,
                    edgecolor='black', linewidth=2,
                    transform=ccrs.PlateCarree(),
                ))

    for j in range(len(datasets), len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout(
        rect=[0, 0.03 if colorbar_orientation == 'horizontal' else 0, 1, 0.92]
    )

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes[:len(datasets)].tolist(),
            orientation=colorbar_orientation,
            fraction=colorbar_fraction, pad=colorbar_pad,
            aspect=colorbar_aspect, extend='both',
        )
        cbar.set_label(colorbar_label, fontsize=12)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


# =============================================================================
# Trend maps
# =============================================================================

def plot_trend_map(
    data: xr.DataArray,
    vmin: float,
    vmax: float,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    cmap: str = 'RdBu_r',
    levels: int = 21,
    projection: ccrs.Projection = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Global map of a single 2D trend field with a horizontal colorbar.

    Parameters
    ----------
    data : xr.DataArray with ``lat`` and ``lon`` coords.
    vmin, vmax : float
    figsize, title, colorbar_label, cmap, levels : standard plotting options.
    projection : cartopy.crs.Projection — default Robinson(central_longitude=180).
    savepath : str, optional
    """
    if projection is None:
        projection = ccrs.Robinson(central_longitude=180)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})

    lons = data['lon'].values
    lats = data['lat'].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    contour_levels = np.linspace(vmin, vmax, levels)
    cf = ax.contourf(
        lon2d, lat2d, data.values,
        levels=contour_levels, cmap=cmap,
        vmin=vmin, vmax=vmax, extend='both',
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.8,
                      color='black', alpha=0.6, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}
    gl.xlocator = plt.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal',
                        pad=0.07, aspect=40, shrink=0.8, extend='both')
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=12)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig


def plot_cmip6_trends(
    trend_data: Dict[str, xr.DataArray],
    vmin: float,
    vmax: float,
    title: str = '',
    colorbar_label: str = '',
    ncols: int = 4,
    figsize_per_panel: Tuple[float, float] = (5, 4),
    cmap: str = 'RdBu_r',
    levels: int = 21,
    projection: ccrs.Projection = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Grid of per-model trend maps sharing one colorbar.

    Parameters
    ----------
    trend_data : dict[str, xr.DataArray]
        Maps model name → trend DataArray on ``lat`` / ``lon``.
    vmin, vmax : float
    title : str
    colorbar_label : str
    ncols : int
    figsize_per_panel : (w, h)
    cmap : str
    levels : int
    projection : cartopy.crs.Projection, default Robinson(central_longitude=180).
    savepath : str, optional
    dpi : int
    """
    if projection is None:
        projection = ccrs.Robinson(central_longitude=180)

    n = len(trend_data)
    if n == 0:
        raise ValueError("trend_data is empty")
    nrows = math.ceil(n / ncols)
    w, h = figsize_per_panel

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(w * ncols, h * nrows),
        subplot_kw={'projection': projection},
    )
    if nrows == 1:
        axes = np.atleast_2d(axes)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    model_names = list(trend_data.keys())
    contour_levels = np.linspace(vmin, vmax, levels)
    cf = None

    for i, model_name in enumerate(model_names):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        da = trend_data[model_name]
        lons = da['lon'].values
        lats = da['lat'].values
        lon2d, lat2d = np.meshgrid(lons, lats)

        cf = ax.contourf(
            lon2d, lat2d, da.values,
            levels=contour_levels, cmap=cmap,
            vmin=vmin, vmax=vmax, extend='both',
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()

        show_labels = (col == 0) or (row == nrows - 1)
        gl = ax.gridlines(draw_labels=show_labels, linewidth=0.5,
                          color='black', alpha=0.5, linestyle='-')
        if show_labels:
            gl.left_labels = (col == 0)
            gl.right_labels = False
            gl.bottom_labels = (row == nrows - 1)
            gl.top_labels = False
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black'}
            gl.xlocator = plt.FixedLocator([-120, -60, 0, 60, 120])
            gl.ylocator = plt.FixedLocator([-60, -30, 0, 30, 60])

        ax.set_title(model_name, fontsize=12)

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=18, y=0.96)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93], pad=2.0)

    cbar = fig.colorbar(
        cf, ax=axes, orientation='horizontal',
        pad=0.02, aspect=50, shrink=0.8, extend='both',
        anchor=(0.5, 0.0), panchor=(0.5, 1.0),
    )
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=14)

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig
