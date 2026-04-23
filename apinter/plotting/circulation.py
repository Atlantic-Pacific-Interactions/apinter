"""Walker / Hadley / velocity-potential panel plots.

Single-axes renderers used by the E3SMS ``_v2`` figure scripts. Callers
build the figure and colorbar; each function here draws one panel with
Gaussian smoothing defaults that match the v2 layouts.
"""
from typing import Optional, Sequence, Tuple

import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import gaussian_filter


def plot_walker_section(ax, psi, lon, plev_hpa, levels,
                        lon_plot_bounds: Tuple[float, float] = (100, 360),
                        cmap: str = 'RdBu_r',
                        smooth_sigma: float = 1.5,
                        contour_label_levels: Optional[Sequence[float]] = None,
                        contour_label_fontsize: int = 13,
                        show_ylabel: bool = False,
                        show_xlabel: bool = False,
                        left_title: str = '',
                        right_title: str = ''):
    """Walker ψ contour plot (longitude × pressure) on a single axes.

    Pressure axis is log-scaled and inverted (surface at bottom). Contour
    levels below 100 hPa are clipped. Returns the ``contourf`` mesh for
    use with ``fig.colorbar``.
    """
    lon = np.asarray(lon)
    plev_hpa = np.asarray(plev_hpa)
    psi = np.asarray(psi)

    lonW, lonE = lon_plot_bounds
    lon_mask = (lon >= lonW) & (lon <= lonE)
    lon_p = lon[lon_mask]
    psi_p = psi[:, lon_mask]

    sort_idx = np.argsort(plev_hpa)[::-1]
    plev_p = plev_hpa[sort_idx]
    psi_p = psi_p[sort_idx, :]
    keep = plev_p >= 100
    plev_p = plev_p[keep]
    psi_p = psi_p[keep, :]

    if smooth_sigma > 0:
        psi_p = gaussian_filter(np.nan_to_num(psi_p, nan=0), sigma=smooth_sigma)

    cf = ax.contourf(lon_p, plev_p, psi_p, levels=levels,
                     cmap=cmap, extend='both')
    cs = ax.contour(lon_p, plev_p, psi_p, levels=levels,
                    colors='k', linewidths=0.4)
    if contour_label_levels is None:
        contour_label_levels = [l for l in np.arange(-2, 2.1, 0.4) if abs(l) > 0.01]
    # Map each requested label level to the nearest actually-drawn contour
    # level (within a small tolerance), to avoid clabel's strict match check.
    cs_levels = np.asarray(cs.levels, dtype=float)
    label_lvls = []
    for lvl in contour_label_levels:
        if cs_levels.size == 0:
            continue
        i = int(np.argmin(np.abs(cs_levels - float(lvl))))
        if np.isclose(cs_levels[i], float(lvl), atol=1e-6):
            label_lvls.append(cs_levels[i])
    if label_lvls:
        ax.clabel(cs, levels=label_lvls,
                  inline=True, fontsize=contour_label_fontsize, fmt='%g')
    ax.contour(lon_p, plev_p, psi_p, levels=[0], colors='k', linewidths=1.5)

    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(['1000', '850', '700', '500', '300', '200', '100'])
    ax.set_xlim(lonW, lonE)

    xt = np.arange(int(np.ceil(lonW / 30) * 30),
                   int(np.floor(lonE / 30) * 30) + 1, 30)
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{x}°E' if x <= 180 else f'{360 - x}°W' for x in xt])

    if left_title:
        ax.set_title(left_title, fontsize=18, fontweight='bold', loc='left')
    if right_title:
        ax.set_title(right_title, fontsize=18, fontweight='bold', loc='right')
    if show_ylabel:
        ax.set_ylabel('Pressure (hPa)', fontsize=15)
    if show_xlabel:
        ax.set_xlabel('Longitude', fontsize=15)
    return cf


def plot_hadley_section(ax, psi, lat, plev_hpa, levels,
                        lat_plot_bounds: Tuple[float, float] = (-30, 30),
                        cmap: str = 'RdBu_r',
                        smooth_sigma: float = 1.0,
                        show_ylabel: bool = False,
                        show_xlabel: bool = False,
                        left_title: str = '',
                        right_title: str = ''):
    """Hadley ψ contour plot (latitude × pressure) on a single axes."""
    lat = np.asarray(lat)
    plev_hpa = np.asarray(plev_hpa)
    psi = np.asarray(psi)

    latS, latN = lat_plot_bounds
    lat_mask = (lat >= latS) & (lat <= latN)
    lat_p = lat[lat_mask]
    psi_p = psi[:, lat_mask]

    sort_idx = np.argsort(plev_hpa)[::-1]
    plev_p = plev_hpa[sort_idx]
    psi_p = psi_p[sort_idx, :]
    keep = plev_p >= 100
    plev_p = plev_p[keep]
    psi_p = psi_p[keep, :]

    if smooth_sigma > 0:
        psi_p = gaussian_filter(np.nan_to_num(psi_p, nan=0), sigma=smooth_sigma)

    cf = ax.contourf(lat_p, plev_p, psi_p, levels=levels,
                     cmap=cmap, extend='both')
    ax.contour(lat_p, plev_p, psi_p, levels=levels, colors='k', linewidths=0.4)
    ax.contour(lat_p, plev_p, psi_p, levels=[0], colors='k', linewidths=1.5)

    ax.set_yscale('log')
    ax.set_ylim(1000, 100)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(['1000', '850', '700', '500', '300', '200', '100'])
    ax.set_xlim(latS, latN)

    if left_title:
        ax.set_title(left_title, fontsize=18, fontweight='bold', loc='left')
    if right_title:
        ax.set_title(right_title, fontsize=18, fontweight='bold', loc='right')
    if show_ylabel:
        ax.set_ylabel('Pressure (hPa)', fontsize=15)
    if show_xlabel:
        ax.set_xlabel('Latitude', fontsize=15)
    return cf


def plot_velpot_panel(ax, chi, u_div, v_div, lat, lon, levels,
                      quiver_step: int = 6,
                      quiver_scale: float = 50.0,
                      min_wind: float = 0.05,
                      cmap: str = 'RdBu_r',
                      left_title: str = '',
                      right_title: str = ''):
    """Velocity-potential contours + divergent-wind quivers on a lat/lon panel.

    Caller is responsible for whatever projection they pass ``ax`` with; by
    default this draws in the axes' native coordinates. Pass ``chi`` scaled
    to 10^6 m²/s if you want readable contour labels.
    """
    cf = ax.contourf(lon, lat, chi, levels=levels, cmap=cmap, extend='both')
    ax.contour(lon, lat, chi, levels=levels, colors='k', linewidths=0.4)

    u2 = np.asarray(u_div)
    v2 = np.asarray(v_div)
    mag = np.sqrt(u2 ** 2 + v2 ** 2)
    weak = mag < min_wind
    u2 = np.where(weak, np.nan, u2)
    v2 = np.where(weak, np.nan, v2)

    lons2d, lats2d = np.meshgrid(lon, lat)
    s = quiver_step
    ax.quiver(lons2d[::s, ::s], lats2d[::s, ::s],
              u2[::s, ::s], v2[::s, ::s],
              scale=quiver_scale, scale_units='width', width=0.0015)

    if left_title:
        ax.set_title(left_title, fontsize=18, fontweight='bold', loc='left')
    if right_title:
        ax.set_title(right_title, fontsize=18, fontweight='bold', loc='right')
    return cf
