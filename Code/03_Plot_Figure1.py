#!/usr/bin/env python
# coding: utf-8

# 1. Read the panel data and set the plotting style

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, box
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FuncFormatter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'

workbook_path = DATA_DIR / '03_Figure1_Source_Data.xlsx'
panel_sheets = pd.read_excel(workbook_path, sheet_name=['Fig1a', 'Fig1b', 'Fig1c', 'Fig1d'])
panel_c = panel_sheets['Fig1c']
panel_d = panel_sheets['Fig1d']
boundary_path = DATA_DIR / 'Map_Boundary' / 'WB_GAD_ADM0_Income.shp'

TARGETS = ['Frequency', 'MBEP']
AREA_ORDER = ['0-1', '1-10', '10-100', '100-1000']
CLIMATE_ORDER = ['Tropical', 'Arid', 'Temperate', 'Cold']
CATEGORY_ORDER = ['Significant increase', 'Not significant', 'Significant decrease']
CATEGORY_COLORS = ['#E64B4B', '#BDC3C7', '#4596DA']
TARGET_COLORS = {'Frequency': '#62BCD0', 'MBEP': '#E98D7D'}
COLOR_LIMITS = {'Frequency': 0.5, 'MBEP': 2.0}
ALPHA = 0.05

palette_rgb = [[36, 0, 216], [24, 28, 247], [40, 87, 255], [61, 135, 255], [86, 176, 255], [117, 211, 255], [153, 234, 255], [188, 249, 255], [234, 255, 255], [255, 255, 234], [255, 241, 188], [255, 214, 153], [255, 172, 117], [255, 120, 86], [255, 61, 61], [247, 39, 53], [216, 21, 47], [165, 0, 33]]
map_cmap = LinearSegmentedColormap.from_list('BlueDarkRed18', np.array(palette_rgb) / 255)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'figure.dpi': 140,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})


# 2. Organize the map data and calculate latitude and area summaries

def classify_ue(effects, adjusted_p):
    significant = adjusted_p <= ALPHA
    increasing = significant & (effects > 0)
    decreasing = significant & (effects < 0)
    categories = np.select(
        [increasing, decreasing],
        ['Significant increase', 'Significant decrease'],
        default='Not significant',
    )
    return categories


def summarize_values(values):
    number = len(values)
    mean = values.mean()
    standard_deviation = values.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(number) if number > 1 else np.nan
    return {'N': number, 'Mean_UE': mean, 'SD': standard_deviation, 'SEM': standard_error}


map_data = {}
latitude_records = []
area_records = []
latitude_edges = np.arange(-60, 85.01, 2.5)

for target, sheet_name in [('Frequency', 'Fig1a'), ('MBEP', 'Fig1b')]:
    frame = panel_sheets[sheet_name].copy()
    frame['UE'] = frame[f'{target}_UE_pp_per_year']
    frame['Q_BH'] = frame[f'{target}_MK_p_BH']
    frame['Category'] = classify_ue(frame['UE'], frame['Q_BH'])
    frame['Area_class'] = pd.cut(
        frame['Lake_area_km2'],
        bins=[0, 1, 10, 100, 1000],
        labels=AREA_ORDER,
        right=False,
    ).astype(str)
    bin_indices = np.searchsorted(latitude_edges, frame['Latitude'], side='right') - 1
    frame['Latitude_bin'] = np.minimum(bin_indices, len(latitude_edges) - 2)
    map_data[target] = frame

    for bin_number in range(len(latitude_edges) - 1):
        values = frame.loc[frame['Latitude_bin'] == bin_number, 'UE']
        lower = latitude_edges[bin_number]
        upper = latitude_edges[bin_number + 1]
        latitude_record = {
            'Target': target,
            'Latitude_lower': lower,
            'Latitude_upper': upper,
            'Latitude_center': (lower + upper) / 2,
            'N': len(values),
            'Mean_UE': values.mean(),
            'P25': values.quantile(0.25),
            'P75': values.quantile(0.75),
        }
        latitude_records.append(latitude_record)

    for area_class in AREA_ORDER:
        values = frame.loc[frame['Area_class'] == area_class, 'UE']
        area_record = {'Target': target, 'Area_class': area_class}
        area_record.update(summarize_values(values))
        area_records.append(area_record)

latitude_summary = pd.DataFrame(latitude_records)
area_summary = pd.DataFrame(area_records)


# 3. Calculate the donut proportions and climate-zone means

inner_records = []
outer_records = []
climate_records = []

for target in TARGETS:
    effects = panel_c[f'{target}_UE_pp_per_year']
    adjusted_p = panel_c[f'{target}_MK_p_BH']
    categories = classify_ue(effects, adjusted_p)
    number_of_groups = len(panel_c)

    for category in CATEGORY_ORDER:
        selected = panel_c.loc[categories == category]
        category_size = len(selected)
        inner_records.append({
            'Target': target,
            'Category': category,
            'N': category_size,
            'Percent_all': 100 * category_size / number_of_groups,
        })
        continent_counts = selected['Continent'].value_counts()
        for continent, count in continent_counts.items():
            outer_records.append({
                'Target': target,
                'Category': category,
                'Continent': continent,
                'N': int(count),
                'Percent_within_category': 100 * count / category_size,
            })

    for climate_zone in CLIMATE_ORDER:
        selected = panel_d.loc[panel_d['Climate_zone'] == climate_zone]
        values = selected[f'{target}_UE_pp_per_year']
        climate_record = {'Target': target, 'Climate_zone': climate_zone}
        climate_record.update(summarize_values(values))
        climate_records.append(climate_record)

donut_inner = pd.DataFrame(inner_records)
donut_outer = pd.DataFrame(outer_records)
climate_summary = pd.DataFrame(climate_records)


# 4. Prepare the offline map and map-drawing function

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


def draw_map(ax, target):
    for path in world_paths:
        land_patch = PathPatch(
            path, facecolor='#F0F0F0', edgecolor='#C7C7C7', linewidth=0.4, zorder=1
        )
        ax.add_patch(land_patch)
    boundary = Polygon(
        map_outline, closed=True, fill=False, edgecolor='#333333', linewidth=1.0, zorder=2
    )
    ax.add_patch(boundary)

    frame = map_data[target]
    limit = COLOR_LIMITS[target]
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    marker_sizes = {'0-1': 11, '1-10': 18, '10-100': 28, '100-1000': 38}
    layers = [
        ('Not significant', 0.4, 3),
        ('Significant decrease', 0.95, 4),
        ('Significant increase', 0.95, 5),
    ]
    for category, opacity, layer in layers:
        selected = frame.loc[frame['Category'] == category]
        sizes = selected['Area_class'].map(marker_sizes)
        ax.scatter(
            selected['Map_x_m'], selected['Map_y_m'],
            c=selected['UE'], s=sizes, cmap=map_cmap, norm=normalization,
            alpha=opacity, linewidths=0, zorder=layer,
        )

    ax.set_xlim(map_outline[:, 0].min(), map_outline[:, 0].max())
    ax.set_ylim(map_outline[:, 1].min(), map_outline[:, 1].max())
    ax.set_aspect('equal')
    ax.axis('off')
    colorbar = ax.figure.colorbar(
        ScalarMappable(norm=normalization, cmap=map_cmap), ax=ax,
        orientation='horizontal', pad=0.065, fraction=0.05, shrink=0.58, extend='both',
    )
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_label(target + r' UE (pp yr$^{-1}$)', labelpad=3)
    colorbar.ax.tick_params(length=3, labelsize=9)


# 5. Draw the latitude profiles and grouped means

def plain_ticks(ax, axis='y'):
    selected_axis = ax.yaxis if axis == 'y' else ax.xaxis
    selected_axis.set_major_locator(MaxNLocator(4))
    formatter = FuncFormatter(
        lambda value, position: f'{value:.3f}'.rstrip('0').rstrip('.') if value else '0'
    )
    selected_axis.set_major_formatter(formatter)


def draw_latitude(ax, target):
    summary = latitude_summary.loc[latitude_summary['Target'] == target].reset_index(drop=True)
    color = TARGET_COLORS[target]
    ax.axvline(0, color='#555555', linestyle='--', linewidth=0.9)
    ax.fill_betweenx(
        summary['Latitude_center'], summary['P25'], summary['P75'],
        color=color, alpha=0.25, linewidth=0,
    )
    ax.plot(summary['Mean_UE'], summary['Latitude_center'], color=color, linewidth=1.5)

    valid_indices = np.flatnonzero(summary['Mean_UE'].notna().to_numpy())
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        if right - left > 1:
            endpoints = summary.iloc[[left, right]]
            ax.plot(
                endpoints['Mean_UE'], endpoints['Latitude_center'],
                color=color, linewidth=1.5, linestyle=(0, (2, 1.5)),
            )
    ax.set_ylim(-60, 85)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel('Mean UE')
    ax.set_ylabel('Latitude (°)')
    plain_ticks(ax, axis='x')


def draw_sem(ax, positions, summary):
    lower = summary['Mean_UE'] - summary['SEM']
    upper = summary['Mean_UE'] + summary['SEM']
    valid = lower.notna() & upper.notna()
    selected_positions = np.asarray(positions)[valid.to_numpy()]
    ax.vlines(selected_positions, lower[valid], upper[valid], color='#555555', linewidth=1, zorder=4)
    ax.hlines(lower[valid], selected_positions - 0.055, selected_positions + 0.055,
              color='#555555', linewidth=1, zorder=4)
    ax.hlines(upper[valid], selected_positions - 0.055, selected_positions + 0.055,
              color='#555555', linewidth=1, zorder=4)


def draw_area(ax, target):
    summary = area_summary.loc[area_summary['Target'] == target]
    summary = summary.set_index('Area_class').loc[AREA_ORDER]
    positions = np.arange(len(AREA_ORDER))
    ax.bar(
        positions, summary['Mean_UE'], width=0.58,
        color=TARGET_COLORS[target], edgecolor='#555555', linewidth=0.8, zorder=3,
    )
    draw_sem(ax, positions, summary)
    ax.axhline(0, color='#333333', linewidth=0.9)
    ax.set_xticks(positions, AREA_ORDER)
    ax.set_xlabel(r'Lake area (km$^2$)')
    ax.set_ylabel('Mean UE')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', color='#E6E6E6', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    plain_ticks(ax)


def draw_climate(ax):
    positions = np.arange(len(CLIMATE_ORDER))
    width = 0.36
    for target, offset in [('Frequency', -width / 2), ('MBEP', width / 2)]:
        summary = climate_summary.loc[climate_summary['Target'] == target]
        summary = summary.set_index('Climate_zone').loc[CLIMATE_ORDER]
        target_positions = positions + offset
        ax.bar(
            target_positions, summary['Mean_UE'], width=width,
            color=TARGET_COLORS[target], edgecolor='#555555',
            linewidth=0.8, label=target, zorder=3,
        )
        draw_sem(ax, target_positions, summary)
    ax.axhline(0, color='#333333', linewidth=0.9)
    ax.set_xticks(positions, ['Trop.', 'Arid', 'Temp.', 'Cold'])
    ax.set_xlabel('Climate zone')
    ax.set_ylabel('Mean UE')
    ax.grid(axis='y', color='#E6E6E6', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=8)
    plain_ticks(ax)
    ax.legend(loc='upper left', fontsize=8.5)


# 6. Draw the nested donut charts

def draw_donut(ax, target):
    inner = donut_inner.loc[donut_inner['Target'] == target]
    inner = inner.set_index('Category').loc[CATEGORY_ORDER]
    outer_counts = []
    outer_colors = []

    for category, category_color in zip(CATEGORY_ORDER, CATEGORY_COLORS):
        selected = donut_outer.loc[
            (donut_outer['Target'] == target) & (donut_outer['Category'] == category)
        ]
        selected = selected.sort_values('N', ascending=False)
        base_rgb = np.array(to_rgb(category_color))
        for number, row in enumerate(selected.itertuples()):
            shade = base_rgb * (0.60 - 0.07 * number) + (0.40 + 0.07 * number)
            outer_counts.append(row.N)
            outer_colors.append(shade)

    ax.pie(
        inner['N'], radius=0.72, colors=CATEGORY_COLORS,
        startangle=90, counterclock=False,
        wedgeprops={'width': 0.30, 'edgecolor': 'white', 'linewidth': 0.9},
    )
    ax.pie(
        outer_counts, radius=1.0, colors=outer_colors,
        startangle=90, counterclock=False,
        wedgeprops={'width': 0.28, 'edgecolor': 'white', 'linewidth': 0.8},
    )
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.60, 1.60)
    ax.set_aspect('equal')
    ax.axis('off')


# 7. Assemble Figure 1

fig = plt.figure(figsize=(13.2, 13.3))

for target, bottom, letter in [('Frequency', 0.72, 'a'), ('MBEP', 0.39, 'b')]:
    map_axis = fig.add_axes([0.025, bottom - 0.01, 0.50, 0.255])
    latitude_axis = fig.add_axes([0.595, bottom, 0.125, 0.25])
    area_axis = fig.add_axes([0.815, bottom, 0.18, 0.25])
    draw_map(map_axis, target)
    draw_latitude(latitude_axis, target)
    draw_area(area_axis, target)
    fig.text(0.012, bottom + 0.267, letter, fontsize=15, fontweight='bold')

frequency_donut_axis = fig.add_axes([0.005, 0.005, 0.365, 0.30])
mbep_donut_axis = fig.add_axes([0.385, 0.005, 0.365, 0.30])
climate_axis = fig.add_axes([0.815, 0.045, 0.18, 0.235])
draw_donut(frequency_donut_axis, 'Frequency')
draw_donut(mbep_donut_axis, 'MBEP')
draw_climate(climate_axis)
fig.text(0.012, 0.31, 'c', fontsize=15, fontweight='bold')
fig.text(0.765, 0.31, 'd', fontsize=15, fontweight='bold')

plt.show()





