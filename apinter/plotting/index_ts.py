"""Index time-series plots.

Ported and generalized from
``/pscratch/sd/y/yanxia/backup/ENSO-CLOUD/CMIP/cmip_plot.py``
(``plot_nat_ts`` and ``plot_cmip_ts_index``).
Works on any already-filtered, standardized index DataArray (e.g. output
from :func:`apinter.indices.calculate_index`).
"""
from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import xarray as xr


def plot_index_ts(data: xr.DataArray,
                  ax: plt.Axes,
                  left_title: Optional[str] = None,
                  center_title: Optional[str] = None,
                  right_title: Optional[str] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  ref_lines: Tuple[float, ...] = (-0.5, 0.5),
                  pos_color: str = 'red',
                  neg_color: str = 'blue',
                  linewidth: float = 2.0,
                  title_fontsize: int = 22,
                  tick_fontsize: int = 20,
                  year_major: int = 5,
                  show_grid: bool = True) -> None:
    """Single-panel index time series with zero-line, reference lines, and
    red/blue positive/negative shading.

    Parameters
    ----------
    data : xr.DataArray
        1D time series with a ``time`` coord.
    ax : matplotlib.axes.Axes
    left_title, center_title, right_title : optional
        Three-part panel titles.
    ylim : (low, high), optional
        Y-axis limits. None → matplotlib default.
    ref_lines : tuple of floats
        Horizontal reference lines to draw (dashed grey).
    pos_color, neg_color : str
        Fill colors for positive and negative excursions.
    linewidth : float
    title_fontsize, tick_fontsize : int
    year_major : int
        Major x-axis tick spacing in years.
    show_grid : bool
    """
    ax.axhline(0, color='grey')
    for y in ref_lines:
        ax.axhline(y, color='grey', linestyle='--', linewidth=1.5)

    ax.plot(data.time.values, data.values, color='black', linewidth=linewidth)
    ax.fill_between(data.time.values, data.values, 0,
                    where=data.values > 0, color=pos_color)
    ax.fill_between(data.time.values, data.values, 0,
                    where=data.values < 0, color=neg_color)

    ax.xaxis.set_major_locator(mdates.YearLocator(year_major))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=tick_fontsize)

    if left_title:
        ax.set_title(left_title, fontsize=title_fontsize, loc='left')
    if center_title:
        ax.set_title(center_title, fontsize=title_fontsize, loc='center')
    if right_title:
        ax.set_title(right_title, fontsize=title_fontsize, loc='right')

    if ylim is not None:
        ax.set_ylim(ylim)
    if show_grid:
        ax.grid(True)


def plot_index_grid(indices: Dict[str, xr.DataArray],
                    ncols: int = 3,
                    figsize_per_panel: Tuple[float, float] = (15, 5),
                    center_title: Optional[str] = None,
                    right_title: Optional[str] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    savepath: Optional[str] = None,
                    dpi: int = 300,
                    **plot_kwargs) -> plt.Figure:
    """Tile ``plot_index_ts`` over a dict of named index time series.

    Parameters
    ----------
    indices : dict[str, xr.DataArray]
        Maps a panel label (used as left title) to a 1D index series.
    ncols : int
    figsize_per_panel : (w, h)
    center_title, right_title : optional
        Shared center / right titles, passed to every panel.
    ylim : (low, high), optional
        Shared y-limits.
    savepath : str, optional
        If given, save the figure to this path.
    dpi : int
    **plot_kwargs
        Passed through to :func:`plot_index_ts`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(indices)
    nrows = (n + ncols - 1) // ncols
    w, h = figsize_per_panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (label, data) in zip(axes, indices.items()):
        plot_index_ts(
            data, ax,
            left_title=label,
            center_title=center_title,
            right_title=right_title,
            ylim=ylim,
            **plot_kwargs,
        )

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    return fig
