#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, box
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FuncFormatter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'

panel_sheets = pd.read_excel(
    DATA_DIR / '04_Figure2_Source_Data.xlsx',
    sheet_name=['Fig2a', 'Fig2b', 'Fig2c', 'Fig2d'],
)
boundary_path = DATA_DIR / 'Map_Boundary' / 'WB_GAD_ADM0_Income.shp'

TARGETS = ['Frequency', 'MBEP']
AREA_ORDER = ['0-1', '1-10', '10-100', '100-1000']
AREA_EDGES = [0, 1, 10, 100, 1000]
EVENT_BAND_MULTIPLIER = 1.96
TARGET_COLORS = {'Frequency': '#62BCD0', 'MBEP': '#E98D7D'}
EVENT_COLOR = '#4DBBD5'
FIT_COLOR = '#E64B35'
palette_rgb = [[36, 0, 216], [24, 28, 247], [40, 87, 255], [61, 135, 255], [86, 176, 255], [117, 211, 255], [153, 234, 255], [188, 249, 255], [234, 255, 255], [255, 255, 234], [255, 241, 188], [255, 214, 153], [255, 172, 117], [255, 120, 86], [255, 61, 61], [247, 39, 53], [216, 21, 47], [165, 0, 33]]
map_cmap = LinearSegmentedColormap.from_list('BlueDarkRed18', np.array(palette_rgb) / 255)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 6,
    'figure.dpi': 220, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


# 2. Calculate the pre/post log-change differences

map_data = {}

for target, sheet_name in [('Frequency', 'Fig2a'), ('MBEP', 'Fig2b')]:
    frame = panel_sheets[sheet_name].copy()
    frame['Area_class'] = pd.cut(
        frame['Lake_area_km2'], bins=AREA_EDGES, labels=AREA_ORDER, right=False
    ).astype(str)

    frame['Delta_r'] = frame['Urban_log_change'] - frame['Reference_log_change']
    map_data[target] = frame


# 3. Summarize the latitude bands and lake-area classes

latitude_edges = np.arange(-60, 85.01, 2.5)
latitude_records = []
area_records = []

for target in TARGETS:
    frame = map_data[target]
    bin_indices = np.searchsorted(latitude_edges, frame['Latitude'], side='right') - 1
    frame['Latitude_bin'] = np.minimum(bin_indices, len(latitude_edges) - 2)

    for index in range(len(latitude_edges) - 1):
        values = frame.loc[frame['Latitude_bin'] == index, 'Delta_r']
        lower = latitude_edges[index]
        upper = latitude_edges[index + 1]
        latitude_records.append({
            'Target': target, 'Latitude_center': (lower + upper) / 2,
            'N': len(values), 'Mean_delta_r': values.mean(),
            'P25': values.quantile(0.25), 'P75': values.quantile(0.75),
        })

    for area_class in AREA_ORDER:
        values = frame.loc[frame['Area_class'] == area_class, 'Delta_r']
        standard_deviation = values.std(ddof=1)
        standard_error = standard_deviation / np.sqrt(len(values)) if len(values) > 1 else np.nan
        area_records.append({
            'Target': target, 'Area_class': area_class, 'N': len(values),
            'Mean_delta_r': values.mean(), 'SD': standard_deviation, 'SEM': standard_error,
        })

latitude_summary = pd.DataFrame(latitude_records)
area_summary = pd.DataFrame(area_records)


# 4. Read the event-aligned curve summaries

event_frames = []
for target, sheet_name in [('Frequency', 'Fig2c'), ('MBEP', 'Fig2d')]:
    frame = panel_sheets[sheet_name].copy()
    frame['Target'] = target
    event_frames.append(frame)
event_summary = pd.concat(event_frames, ignore_index=True)

band_width = EVENT_BAND_MULTIPLIER * event_summary['SEM_pp']
event_summary['Band_lower_pp'] = event_summary['Mean_difference_pp'] - band_width
event_summary['Band_upper_pp'] = event_summary['Mean_difference_pp'] + band_width


# 5. Fit the pre- and post-expansion segments

fit_records = []
for target in TARGETS:
    for area_class in AREA_ORDER:
        group = event_summary.loc[
            event_summary['Target'].eq(target) & event_summary['Area_class'].eq(area_class)
        ]
        for phase, first, last in [('Pre', -10, -1), ('Post', 1, 10)]:
            selected = group.loc[
                group['Relative_year'].between(first, last)
                & group['Mean_difference_pp'].notna()
            ]
            if len(selected) >= 2:
                slope, intercept = np.polyfit(
                    selected['Relative_year'], selected['Mean_difference_pp'], 1
                )
            else:
                slope, intercept = np.nan, np.nan
            fit_records.append({
                'Target': target, 'Area_class': area_class, 'Phase': phase,
                'N_relative_years': len(selected),
                'Slope_pp_per_year': slope, 'Intercept_pp': intercept,
            })
fit_summary = pd.DataFrame(fit_records)


# ## 6. Prepare the local shapefile and projected coordinates

source_crs = CRS.from_wkt(boundary_path.with_suffix('.prj').read_text())
projection = Transformer.from_crs(
    source_crs, '+proj=robin +datum=WGS84 +units=m', always_xy=True
)
for frame in map_data.values():
    frame['Map_x_m'], frame['Map_y_m'] = projection.transform(
        frame['Longitude'].to_numpy(), frame['Latitude'].to_numpy()
    )

map_extent = box(-180, -60, 180, 85)
world_paths = []

with shapefile.Reader(str(boundary_path)) as boundary_reader:
    for boundary_shape in boundary_reader.iterShapes():
        boundary = shape(boundary_shape.__geo_interface__)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)
        boundary = boundary.intersection(map_extent)
        boundary = boundary.simplify(0.035, preserve_topology=True)

        if boundary.geom_type == 'Polygon':
            parts = [boundary]
        elif hasattr(boundary, 'geoms'):
            parts = list(boundary.geoms)
        else:
            parts = []

        for polygon in parts:
            if polygon.geom_type != 'Polygon' or polygon.is_empty or polygon.area < 0.005:
                continue
            vertices = []
            path_codes = []
            for ring in [polygon.exterior, *polygon.interiors]:
                coordinates = np.asarray(ring.coords)
                x, y = projection.transform(coordinates[:, 0], coordinates[:, 1])
                ring_vertices = np.column_stack([x, y])
                ring_codes = [MplPath.MOVETO]
                ring_codes.extend([MplPath.LINETO] * (len(ring_vertices) - 2))
                ring_codes.append(MplPath.CLOSEPOLY)
                vertices.extend(ring_vertices)
                path_codes.extend(ring_codes)
            world_paths.append(MplPath(vertices, path_codes))

outline_longitudes = np.concatenate([
    np.linspace(-180, 180, 361), np.full(146, 180),
    np.linspace(180, -180, 361), np.full(146, -180),
])
outline_latitudes = np.concatenate([
    np.full(361, -60), np.linspace(-60, 85, 146),
    np.full(361, 85), np.linspace(85, -60, 146),
])
outline_x, outline_y = projection.transform(outline_longitudes, outline_latitudes)
map_outline = np.column_stack([outline_x, outline_y])


# 7. Draw the maps, latitude profiles and area bars

def plain_ticks(ax, axis='y'):
    selected_axis = ax.yaxis if axis == 'y' else ax.xaxis
    selected_axis.set_major_locator(MaxNLocator(4))
    selected_axis.set_major_formatter(FuncFormatter(
        lambda value, position: f'{value:.3f}'.rstrip('0').rstrip('.') if value else '0'
    ))


def draw_map(ax, target):
    for path in world_paths:
        ax.add_patch(PathPatch(
            path, facecolor='#F0F0F0', edgecolor='#C7C7C7', linewidth=0.25, zorder=1
        ))
    ax.add_patch(Polygon(
        map_outline, closed=True, fill=False, edgecolor='#333333', linewidth=0.65, zorder=2
    ))
    frame = map_data[target]
    marker_sizes = {'0-1': 4, '1-10': 7, '10-100': 11, '100-1000': 15}
    normalization = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    for positive, layer in [(False, 3), (True, 4)]:
        selected = frame.loc[frame['Delta_r'].ge(0).eq(positive)]
        ax.scatter(
            selected['Map_x_m'], selected['Map_y_m'], c=selected['Delta_r'],
            s=selected['Area_class'].map(marker_sizes), cmap=map_cmap, norm=normalization,
            alpha=0.95, linewidths=0, zorder=layer,
        )
    ax.set_xlim(map_outline[:, 0].min(), map_outline[:, 0].max())
    ax.set_ylim(map_outline[:, 1].min(), map_outline[:, 1].max())
    ax.set_aspect('equal')
    ax.axis('off')
    colorbar = ax.figure.colorbar(
        ScalarMappable(norm=normalization, cmap=map_cmap), ax=ax,
        orientation='horizontal', pad=0.055, fraction=0.045, shrink=0.72, extend='both',
    )
    colorbar.set_ticks([-2, 0, 2])
    colorbar.ax.tick_params(length=2, labelsize=6)
    colorbar.set_label(target + r': log-change difference ($\Delta r$)', labelpad=2)


def draw_latitude(ax, target):
    summary = latitude_summary.loc[latitude_summary['Target'] == target].reset_index(drop=True)
    ax.axvline(0, color='#555555', linewidth=0.65, linestyle='--')
    ax.fill_betweenx(
        summary['Latitude_center'], summary['P25'], summary['P75'],
        color=TARGET_COLORS[target], alpha=0.25, linewidth=0,
    )
    ax.plot(summary['Mean_delta_r'], summary['Latitude_center'], color=TARGET_COLORS[target], linewidth=1.1)
    valid_indices = np.flatnonzero(summary['Mean_delta_r'].notna().to_numpy())
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        if right - left > 1:
            endpoints = summary.iloc[[left, right]]
            ax.plot(
                endpoints['Mean_delta_r'], endpoints['Latitude_center'],
                color=TARGET_COLORS[target], linewidth=1.1, linestyle=(0, (2, 1.5)),
            )
    ax.set_ylim(-60, 85)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel(r'Mean $\Delta r$')
    ax.set_ylabel('Latitude (°)')
    ax.tick_params(length=2)
    plain_ticks(ax, 'x')


def draw_area(ax, target):
    summary = area_summary.loc[area_summary['Target'] == target].set_index('Area_class').loc[AREA_ORDER]
    positions = np.arange(len(AREA_ORDER))
    ax.bar(
        positions, summary['Mean_delta_r'], width=0.58,
        color=TARGET_COLORS[target], edgecolor='#555555', linewidth=0.5, zorder=3,
    )
    valid = summary['SEM'].notna()
    ax.errorbar(
        positions[valid], summary.loc[valid, 'Mean_delta_r'], yerr=summary.loc[valid, 'SEM'],
        fmt='none', ecolor='#555555', elinewidth=0.65, capsize=2, capthick=0.65, zorder=4,
    )
    ax.axhline(0, color='#333333', linewidth=0.7)
    ax.set_xticks(positions, AREA_ORDER)
    ax.set_xlabel(r'Lake area (km$^2$)')
    ax.set_ylabel(r'Mean $\Delta r$')
    ax.grid(axis='y', color='#E6E6E6', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, labelsize=6)
    plain_ticks(ax)


# 8. Draw the event-aligned curves

def draw_event(ax, target, area_class, legend=False):
    summary = event_summary.loc[
        event_summary['Target'].eq(target) & event_summary['Area_class'].eq(area_class)
    ]
    fits = fit_summary.loc[
        fit_summary['Target'].eq(target) & fit_summary['Area_class'].eq(area_class)
    ]
    ax.axvspan(-1, 1, color='#E5E7E9', alpha=0.65, zorder=0)
    ax.axvline(0, color='#555555', linewidth=0.6, linestyle='--', zorder=1)
    ax.axhline(0, color='#555555', linewidth=0.55, zorder=1)
    ax.fill_between(
        summary['Relative_year'], summary['Band_lower_pp'], summary['Band_upper_pp'],
        color=EVENT_COLOR, alpha=0.25, linewidth=0, zorder=2,
    )
    ax.plot(
        summary['Relative_year'], summary['Mean_difference_pp'],
        color=EVENT_COLOR, linewidth=0.9, marker='o', markersize=1.7,
        markeredgecolor='white', markeredgewidth=0.25, zorder=3, label='Mean difference',
    )
    for phase, first, last, text_x in [('Pre', -10, -1, 0.03), ('Post', 1, 10, 0.53)]:
        fit = fits.loc[fits['Phase'] == phase].iloc[0]
        if np.isfinite(fit['Slope_pp_per_year']):
            selected = summary.loc[
                summary['Relative_year'].between(first, last)
                & summary['Mean_difference_pp'].notna()
            ]
            fitted_values = fit['Slope_pp_per_year'] * selected['Relative_year'] + fit['Intercept_pp']
            ax.plot(
                selected['Relative_year'], fitted_values, color=FIT_COLOR,
                linewidth=1, linestyle='--', zorder=4,
                label='Fitted line' if phase == 'Pre' else None,
            )
            slope_text = f"{fit['Slope_pp_per_year']:.4f}".rstrip('0').rstrip('.')
            ax.text(
                text_x, 0.035, f'{phase}: {slope_text}', transform=ax.transAxes,
                color=FIT_COLOR, fontsize=5, ha='left', va='bottom', zorder=5,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 0.7},
            )
    ax.set_title(area_class.replace('-', '–') + r' km$^2$', fontsize=5.8, pad=3)
    ax.set_xlim(-10, 10)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.margins(y=0.18)
    ax.tick_params(length=2, labelsize=5.5)
    plain_ticks(ax)
    if legend:
        ax.legend(loc='upper left', fontsize=4.8, handlelength=1.3, labelspacing=0.2)


# 9. Assemble Figure 2

fig = plt.figure(figsize=(183 / 25.4, 195 / 25.4))

for target, bottom, letter in [('Frequency', 0.725, 'a'), ('MBEP', 0.425, 'b')]:
    map_axis = fig.add_axes([0.025, bottom, 0.495, 0.24])
    latitude_axis = fig.add_axes([0.59, bottom + 0.02, 0.125, 0.235])
    area_axis = fig.add_axes([0.80, bottom + 0.02, 0.185, 0.235])
    draw_map(map_axis, target)
    draw_latitude(latitude_axis, target)
    draw_area(area_axis, target)
    area_axis.tick_params(axis='x', labelsize=5.3)
    fig.text(0.012, bottom + 0.267, letter, fontweight='bold', fontsize=10)

for target, left, letter in [('Frequency', 0.065, 'c'), ('MBEP', 0.575, 'd')]:
    for index, area_class in enumerate(AREA_ORDER):
        row, column = divmod(index, 2)
        event_axis = fig.add_axes([left + column * 0.22, 0.225 - row * 0.17, 0.175, 0.125])
        draw_event(event_axis, target, area_class, legend=index == 0)
        if column == 0:
            event_axis.set_ylabel(r'$\Delta$ ' + target + ' (pp)', fontsize=6, labelpad=2)
    fig.text(
        left + 0.1975, 0.009, 'Years relative to the identified expansion year',
        ha='center', va='bottom', fontsize=6,
    )
    fig.text(left - 0.05, 0.375, letter, fontweight='bold', fontsize=10)

plt.show()




