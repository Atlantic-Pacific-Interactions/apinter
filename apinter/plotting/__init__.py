"""Plotting helpers.

Scope (phase 4):

  - index_ts.py       plot_index_ts, plot_index_grid
  - spatial_maps.py   plot_regression_map, plot_multiple_regression_maps,
                      plot_trend_map, plot_cmip6_trends
  - circulation.py    plot_walker_section, plot_hadley_section, plot_velpot_panel
  - omega_regression.py  plot_omega_lead_lag_profile, create_omega_regression_plots

All functions return the Figure/axes or contour object rather than calling
``plt.show()``; hardcoded save paths are gone — pass ``savepath=...`` (or
``output_dir=...`` for multi-file helpers) when you want a file on disk.
"""
from .index_ts import plot_index_ts, plot_index_grid
from .spatial_maps import (
    plot_regression_map,
    plot_multiple_regression_maps,
    plot_trend_map,
    plot_cmip6_trends,
)
from .circulation import (
    plot_walker_section,
    plot_hadley_section,
    plot_velpot_panel,
)
from .omega_regression import (
    plot_omega_lead_lag_profile,
    create_omega_regression_plots,
)

__all__ = [
    # index
    "plot_index_ts",
    "plot_index_grid",
    # spatial maps
    "plot_regression_map",
    "plot_multiple_regression_maps",
    "plot_trend_map",
    "plot_cmip6_trends",
    # circulation panels
    "plot_walker_section",
    "plot_hadley_section",
    "plot_velpot_panel",
    # omega regression
    "plot_omega_lead_lag_profile",
    "create_omega_regression_plots",
]
