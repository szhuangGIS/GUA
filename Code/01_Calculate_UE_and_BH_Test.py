#!/usr/bin/env python
# coding: utf-8

# 1. Read the covariate data

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'

distribution_path = DATA_DIR / '01_Covariate_Distributions.xlsx'
distributions = pd.read_excel(distribution_path, sheet_name='Data')

VARIABLES = ['ln_area', 'ln_depth', 'elevation', 'ln_N', 'ln_P']
NAMES = ['Lake area', 'Mean depth', 'Elevation', 'N background', 'P background']
X_LABELS = ['ln(area / km²)', 'ln(depth / m)', 'Elevation (m)',
            'ln(N background)', 'ln(P background)']
POPULATIONS = ['Urban', 'Reference_before', 'Reference_after']


# 2. Calculate weighted cumulative distributions and moments

def weighted_ecdf(frame):
    grouped_values = frame.groupby('Value', as_index=False)
    combined_weights = grouped_values['Weight'].sum()
    curve = combined_weights.sort_values('Value')

    weights = curve['Weight']
    cumulative_weights = weights.cumsum()
    total_weight = weights.sum()
    cumulative_fraction = cumulative_weights / total_weight
    curve['CDF'] = cumulative_fraction
    return curve


def weighted_moments(frame):
    values = frame['Value'].to_numpy()
    weights = frame['Weight'].to_numpy()
    mean = np.average(values, weights=weights)

    deviations = values - mean
    squared_deviations = deviations ** 2
    variance = np.average(squared_deviations, weights=weights)
    return mean, variance


curves = {}
moments = {}
for variable in VARIABLES:
    variable_mask = distributions['Variable'].eq(variable)
    for population in POPULATIONS:
        population_mask = distributions['Population'].eq(population)
        selected_rows = variable_mask & population_mask
        subset = distributions.loc[selected_rows]

        curve = weighted_ecdf(subset)
        mean, variance = weighted_moments(subset)
        distribution_key = (variable, population)
        curves[distribution_key] = curve
        moments[distribution_key] = (mean, variance)


# 3. Calculate standardized mean differences

balance_rows = []
selection_stages = [
    ('Before', 'Reference_before'),
    ('After', 'Reference_after'),
]

for variable in VARIABLES:
    urban_mean, urban_var = moments[variable, 'Urban']
    _, before_var = moments[variable, 'Reference_before']
    pooled_variance = (urban_var + before_var) / 2
    pooled_sd = np.sqrt(pooled_variance)
    urban_sd = np.sqrt(urban_var)

    for stage, population in selection_stages:
        reference_mean, reference_var = moments[variable, population]
        reference_sd = np.sqrt(reference_var)
        mean_difference = urban_mean - reference_mean
        signed_smd = mean_difference / pooled_sd
        absolute_smd = abs(signed_smd)

        balance_record = {
            'Variable': variable,
            'Stage': stage,
            'Urban_mean': urban_mean,
            'Reference_mean': reference_mean,
            'Urban_SD': urban_sd,
            'Reference_SD': reference_sd,
            'Baseline_pooled_SD': pooled_sd,
            'SMD': signed_smd,
            'Abs_SMD': absolute_smd,
        }
        balance_rows.append(balance_record)

balance = pd.DataFrame(balance_rows)
smd = balance.pivot(index='Variable', columns='Stage', values='Abs_SMD')
smd = smd.reindex(VARIABLES)
display_names = dict(zip(VARIABLES, NAMES))
smd_display = smd[['Before', 'After']].rename(index=display_names).round(6)
display(smd_display)


# 4. Plot

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
})
BLUE, ORANGE, GREY, DARK = '#397A99', '#B67955', '#C5CDD1', '#36434C'
fig, axes = plt.subplots(2, 3, figsize=np.array([183, 140]) / 25.4)
fig.subplots_adjust(left=0.10, right=0.975, top=0.91, bottom=0.20, hspace=0.60, wspace=0.65)

curve_styles = [
    ('Reference_before', ORANGE, '--'),
    ('Reference_after', BLUE, '-'),
    ('Urban', DARK, ':'),
]
for i, variable in enumerate(VARIABLES):
    ax = axes.flat[i]
    for population, color, style in curve_styles:
        curve = curves[variable, population]
        ax.step(curve['Value'], curve['CDF'], where='post', color=color, ls=style, lw=1.2)
    ax.set_title(NAMES[i], loc='left', pad=12)
    ax.set_xlabel(X_LABELS[i])
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    if i % 3 == 0:
        ax.set_ylabel('Weighted cumulative fraction')

ax = axes[1, 2]
for j, variable in enumerate(VARIABLES):
    before, after = smd.loc[variable, ['Before', 'After']]
    ax.plot([before, after], [j, j], color=GREY, lw=2)
    ax.scatter(before, j, s=24, color=ORANGE, marker='s', zorder=3)
    ax.scatter(after, j, s=24, color=BLUE, marker='o', zorder=4)
    for value, offset, color, alignment in [(before, 3, ORANGE, 'bottom'), (after, -3, BLUE, 'top')]:
        ax.annotate(f'{value:.3f}', (value, j), xytext=(0, offset), textcoords='offset points',
                    ha='left' if value < 0.03 else 'center', va=alignment, fontsize=6.5, color=color)
ax.set_xlim(-0.025, 0.43)
ax.set_ylim(4.65, -0.65)
ax.set_xticks(np.arange(0, 0.41, 0.1))
ax.set_yticks(np.arange(len(VARIABLES)), NAMES)
ax.set_xlabel('|SMD|')
ax.set_title('Covariate balance', loc='left', pad=12)

for letter, ax in zip('abcdef', axes.flat):
    ax.text(-0.14, 1.17, letter, transform=ax.transAxes,
            weight='bold', fontsize=10, va='top', ha='left')
    ax.minorticks_off()
    lo, hi = sorted(ax.get_xlim())
    ax.set_xticks([tick for tick in ax.get_xticks() if lo <= tick <= hi])

curve_handles = [
    Line2D([], [], color=DARK, ls=':', label='Urban (same before / after)'),
    Line2D([], [], color=ORANGE, ls='--', label='Reference before'),
    Line2D([], [], color=BLUE, ls='-', label='Reference after'),
]
balance_handles = [
    Line2D([], [], marker='s', color=ORANGE, ls='', label='Before selection'),
    Line2D([], [], marker='o', color=BLUE, ls='', label='After selection'),
]
fig.legend(handles=curve_handles, loc='lower center', bbox_to_anchor=(0.53, 0.075), ncol=3)
fig.legend(handles=balance_handles, loc='lower center', bbox_to_anchor=(0.53, 0.025), ncol=2)

plt.show()


# 5. Calculate urbanization effects

slope_path = DATA_DIR / '01_UE_Matched_Groups.xlsx'
data = pd.read_excel(slope_path, sheet_name='Data')
INDICATORS = ['Frequency', 'MBEP']

for indicator in INDICATORS:
    urban_column = f'{indicator}_Urban_slope_pp_per_year'
    reference_column = f'{indicator}_Reference_slope_pp_per_year'
    effect_column = f'{indicator}_UE_pp_per_year'

    urban_slopes = data[urban_column]
    reference_slopes = data[reference_column]
    urbanization_effect = urban_slopes - reference_slopes
    data[effect_column] = urbanization_effect


# 6. Apply BH correction

def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    one_dimensional = p.ndim == 1
    finite_values = np.isfinite(p).all()
    within_range = not np.any((p < 0) | (p > 1))
    if not one_dimensional or not finite_values or not within_range:
        raise ValueError('Provide a one-dimensional array of finite p-values between 0 and 1.')

    number_of_tests = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, number_of_tests + 1)
    scaled_p = sorted_p * number_of_tests
    adjusted_sorted = scaled_p / ranks

    reverse_adjusted = adjusted_sorted[::-1]
    reverse_minimum = np.minimum.accumulate(reverse_adjusted)
    monotonic_adjusted = reverse_minimum[::-1]
    bounded_adjusted = np.minimum(monotonic_adjusted, 1.0)

    adjusted = np.empty_like(p)
    adjusted[order] = bounded_adjusted
    return adjusted


for indicator in INDICATORS:
    raw_p_column = f'{indicator}_MK_p'
    adjusted_p_column = f'{indicator}_MK_p_BH'
    raw_p_values = data[raw_p_column]
    adjusted_p_values = benjamini_hochberg(raw_p_values)
    data[adjusted_p_column] = adjusted_p_values


# 7. Summarize significance using UE direction

alpha = 0.05
summary_rows = []
number_of_groups = len(data)

for indicator in INDICATORS:
    adjusted_p_column = f'{indicator}_MK_p_BH'
    effect_column = f'{indicator}_UE_pp_per_year'
    adjusted_p = data[adjusted_p_column]
    ue = data[effect_column]

    significant = adjusted_p <= alpha
    positive_effect = ue > 0
    negative_effect = ue < 0
    significant_increase = significant & positive_effect
    significant_decrease = significant & negative_effect
    not_significant = ~significant

    counts = {
        'Significant increase': significant_increase.sum(),
        'Significant decrease': significant_decrease.sum(),
        'Not significant': not_significant.sum(),
    }

    for category, count in counts.items():
        percentage = 100 * count / number_of_groups
        summary_row = [indicator, category, int(count), percentage]
        summary_rows.append(summary_row)

summary_columns = ['Indicator', 'Category', 'N', 'Percent (%)']
results = pd.DataFrame(summary_rows, columns=summary_columns)
percent_format = {'Percent (%)': '{:.2f}'.format}
summary_text = results.to_string(index=False, formatters=percent_format)
print(summary_text)


# In[ ]:




