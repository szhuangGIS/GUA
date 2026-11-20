#!/usr/bin/env python
# coding: utf-8

# 1. Read the panel data and set plotting parameters

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'
panel_sheets = pd.read_excel(
    DATA_DIR / '06_Figure4_Source_Data.xlsx',
    sheet_name=['Fig4a', 'Fig4b', 'Fig4c'],
)

TARGETS = ['Frequency', 'MBEP']
FEATURES = [
    'Temperature_UE', 'Wind_speed_UE', 'VPD_UE', 'Precipitation_UE',
    'Radiation_UE', 'Soil_moisture_UE', 'LAI_UE', 'Runoff_UE', 'NTL_UE',
]
FEATURE_LABELS = dict(zip(FEATURES, [
    'Temperature', 'Wind speed', 'VPD', 'Precipitation',
    'Radiation', 'Soil moisture', 'LAI', 'Runoff', 'NTL',
]))
COLORS = {'Frequency': '#40B7D0', 'MBEP': '#EF5746'}
FIT_COLORS = {'Frequency': '#087F97', 'MBEP': '#B62D20'}
VIEW_QUANTILES = [0.01, 0.99]
NTL_XLIM = (-1.0, 1.0)
LOWESS_FRAC = 0.25

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.dpi': 190, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


# 2. Calculate relative importance for panel a

importance = panel_sheets['Fig4a'].copy()
total_importance = importance.groupby('Target')['Mean_abs_SHAP_pp_per_year'].transform('sum')
importance['Relative_percent'] = 100 * importance['Mean_abs_SHAP_pp_per_year'] / total_importance
feature_order = (
    importance.groupby('Feature')['Relative_percent']
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
importance_by_target = {}
for target in TARGETS:
    target_rows = importance.loc[importance['Target'].eq(target)]
    importance_by_target[target] = target_rows.set_index('Feature').loc[feature_order]


# 3. Standardize predictors and select the displayed ranges

point_frames = []
for target, sheet_name in [('Frequency', 'Fig4b'), ('MBEP', 'Fig4c')]:
    frame = panel_sheets[sheet_name].copy()
    frame['Target'] = target
    point_frames.append(frame)
points = pd.concat(point_frames, ignore_index=True)

scaling_rows = []
for feature in FEATURES:
    feature_values = panel_sheets['Fig4b'].loc[
        panel_sheets['Fig4b']['Feature'].eq(feature), 'Feature_UE'
    ].dropna()
    mean_value = feature_values.mean()
    sd_value = feature_values.std(ddof=1)
    standardized_values = (feature_values - mean_value) / sd_value
    lower_z, upper_z = standardized_values.quantile(VIEW_QUANTILES).to_numpy()
    if feature == 'NTL_UE':
        lower_z, upper_z = NTL_XLIM
    scaling_rows.append({
        'Feature': feature, 'Mean': mean_value, 'SD': sd_value,
        'Lower_z': lower_z, 'Upper_z': upper_z,
    })
scaling = pd.DataFrame(scaling_rows).set_index('Feature')

points['Feature_z'] = (
    points['Feature_UE'] - points['Feature'].map(scaling['Mean'])
) / points['Feature'].map(scaling['SD'])
points['In_view'] = (
    points['Feature_z'].ge(points['Feature'].map(scaling['Lower_z']))
    & points['Feature_z'].le(points['Feature'].map(scaling['Upper_z']))
)
display_points = points.loc[points['In_view']].copy()


# 4. Fit the SHAP dependence curves

curve_frames = []
for target in TARGETS:
    for feature in FEATURES:
        selected = display_points['Target'].eq(target) & display_points['Feature'].eq(feature)
        sample = display_points.loc[selected].sort_values('Feature_z')
        x_values = sample['Feature_z'].to_numpy()
        shap_values = sample['SHAP_pp_per_year'].to_numpy()
        evaluation_grid = np.unique(np.quantile(x_values, np.linspace(0, 1, 201)))
        fitted_values = lowess(
            shap_values, x_values, frac=LOWESS_FRAC, it=0,
            xvals=evaluation_grid, is_sorted=True, missing='none',
        )
        curve_frames.append(pd.DataFrame({
            'Target': target, 'Feature': feature,
            'Feature_z': evaluation_grid, 'Fitted_SHAP': fitted_values,
        }))
shap_curves = pd.concat(curve_frames, ignore_index=True)

y_limits = {}
for target in TARGETS:
    values = display_points.loc[display_points['Target'].eq(target), 'SHAP_pp_per_year']
    lower_value = min(0, values.min())
    upper_value = max(0, values.max())
    span = upper_value - lower_value
    y_limits[target] = (lower_value - 0.12 * span, upper_value + 0.23 * span)


# 5. Define the importance and dependence panels

def draw_importance(ax):
    positions = np.arange(len(feature_order))
    for target, shift, marker in [('Frequency', -0.10, 'o'), ('MBEP', 0.10, 'D')]:
        values = importance_by_target[target]['Relative_percent']
        ax.vlines(positions + shift, 0, values, color=COLORS[target], alpha=0.25, lw=1.3)
        ax.scatter(
            positions + shift, values, s=26, marker=marker,
            color=COLORS[target], edgecolors='white', linewidth=0.45,
            label=target, zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in feature_order])
    ax.set_ylabel('Relative SHAP\nimportance (%)')
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, importance['Relative_percent'].max() * 1.24)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis='y', color='#E6E6E6', lw=0.5, zorder=0)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.legend(loc='upper right', ncol=2, columnspacing=1.5)


def draw_dependence_grid(axes, target, panel_letter):
    for index, (ax, feature) in enumerate(zip(axes.flat, FEATURES)):
        selected = display_points['Target'].eq(target) & display_points['Feature'].eq(feature)
        sample = display_points.loc[selected]
        curve = shap_curves.loc[
            shap_curves['Target'].eq(target) & shap_curves['Feature'].eq(feature)
        ]
        lower_z, upper_z = scaling.loc[feature, ['Lower_z', 'Upper_z']]
        width = upper_z - lower_z
        ax.scatter(
            sample['Feature_z'], sample['SHAP_pp_per_year'], s=3,
            color=COLORS[target], alpha=0.22, edgecolors='none',
            rasterized=True, zorder=2,
        )
        ax.plot(
            curve['Feature_z'], curve['Fitted_SHAP'],
            color=FIT_COLORS[target], lw=1.35, zorder=4,
        )
        ax.set_xlim(lower_z - 0.035 * width, upper_z + 0.035 * width)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if feature == 'NTL_UE':
            ax.set_xlim(*NTL_XLIM)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(['−1', '−0.5', '0', '0.5', '1'])
        ax.set_ylim(*y_limits[target])
        ax.axhline(0, color='#888888', ls='--', lw=0.55, zorder=1)
        if lower_z < 0 < upper_z:
            ax.axvline(0, color='#CCCCCC', ls=':', lw=0.55, zorder=0)
        ax.set_title(FEATURE_LABELS[feature], pad=5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if index % 3 == 0:
            ax.set_ylabel('SHAP value (pp yr$^{-1}$)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        if index // 3 == 2:
            label = 'NTL UE (z-score)' if feature == 'NTL_UE' else 'Feature UE (z-score)'
            ax.set_xlabel(label, labelpad=3)
    first_ax = axes.flat[0]
    first_ax.text(
        -0.14, 1.15, panel_letter, transform=first_ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom',
    )


# 6. Assemble Figure 4

fig = plt.figure(figsize=(9.8, 7.5))
importance_ax = fig.add_axes([0.07, 0.805, 0.905, 0.16])
draw_importance(importance_ax)
importance_ax.text(
    -0.052, 1.025, 'a', transform=importance_ax.transAxes,
    fontsize=13, fontweight='bold', va='bottom',
)

grids = {
    'Frequency': fig.add_gridspec(
        3, 3, left=0.065, right=0.48, bottom=0.15, top=0.70,
        wspace=0.18, hspace=0.43,
    ),
    'MBEP': fig.add_gridspec(
        3, 3, left=0.575, right=0.99, bottom=0.15, top=0.70,
        wspace=0.18, hspace=0.43,
    ),
}
dependence_axes = {}
for target, title_x, letter in [('Frequency', 0.2725, 'b'), ('MBEP', 0.7825, 'c')]:
    grid = grids[target]
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(3)])
    dependence_axes[target] = axes
    draw_dependence_grid(axes, target, letter)
    fig.text(title_x, 0.747, target, ha='center', color=COLORS[target], fontsize=10)

plt.show()




