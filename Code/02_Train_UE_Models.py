#!/usr/bin/env python
# coding: utf-8

# 1. Load and organize samples

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split, ParameterSampler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from IPython.display import display

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / 'Data' / '02_SHAP_Analysis.xlsx'
data = pd.read_excel(data_path, sheet_name='Data')
data = data.sort_values('Urban_lake_id')
data = data.reset_index(drop=True)

FEATURES = [
    'Temperature_UE', 'NTL_UE', 'VPD_UE', 'Precipitation_UE', 'Runoff_UE',
    'Wind_speed_UE', 'Radiation_UE', 'Soil_moisture_UE', 'LAI_UE',
]
TARGETS = {'Frequency': 'Frequency_UE', 'MBEP': 'MBEP_UE'}
SEED = 20260918
PARTITIONS = {'Original split': 20260918, 'New split': 20261007}
OUTER_FOLDS = 5
INNER_FOLDS = 3
EARLY_STOPPING_ROUNDS = 60
PERMUTATIONS = 20
MODEL_THREADS = 2
OUTER_WORKERS = 4  # Set to 1 to run outer models sequentially.

lake_ids = data['Urban_lake_id']
response_columns = list(TARGETS.values())
response_values = data[response_columns].to_numpy()
predictor_values = data[FEATURES].to_numpy()

if lake_ids.isna().any() or lake_ids.duplicated().any():
    raise ValueError('Urban lake IDs must be present and unique.')
if not np.isfinite(response_values).all():
    raise ValueError('Both response columns must contain finite UE values.')

sample_preview = data.head()
display(sample_preview)


# 2. Specify the candidate parameter configurations

baseline_lambdas = [10., 30., 10., 30., 1., 1., 30., 1., 10., 30., 1., 1.]
baseline_candidates = []

for leaves in [5, 12, 25, 40]:
    for child in [10, 30, 70]:
        candidate_number = len(baseline_candidates)
        regularization = baseline_lambdas[candidate_number]
        configuration = {
            'missing': 'median',
            'n_estimators': 1600,
            'learning_rate': 0.035,
            'num_leaves': leaves,
            'min_child_samples': child,
            'reg_lambda': regularization,
            'colsample_bytree': 1.0,
        }
        baseline_candidates.append(configuration)

search_space = {
    'missing': ['median', 'native'],
    'n_estimators': [1800],
    'learning_rate': [0.015, 0.025, 0.04, 0.06],
    'num_leaves': [5, 8, 12, 20, 31, 50],
    'max_depth': [-1, 3, 5, 7],
    'min_child_samples': [5, 15, 30, 60, 100],
    'reg_alpha': [0., 0.01, 0.1, 1.],
    'reg_lambda': [0., 1., 5., 20., 50.],
    'colsample_bytree': [0.75, 0.9, 1.],
    'subsample': [0.75, 0.9, 1.],
    'subsample_freq': [1],
    'min_split_gain': [0., 0.001, 0.01],
    'extra_trees': [False, True],
}

parameter_sampler = ParameterSampler(
    search_space,
    n_iter=61,
    random_state=20260924,
)
sampled_candidates = list(parameter_sampler)
candidates = baseline_candidates + sampled_candidates

candidate_table = pd.DataFrame(candidates)
candidate_table.index.name = 'Candidate'
candidate_table = candidate_table.reset_index()
print(f'{len(data)} groups; {len(FEATURES)} predictors; {len(candidates)} candidate configurations.')


# 3. Define preprocessing, model fitting and inner validation

def evaluate(observed, predicted):
    coefficient = r2_score(observed, predicted)
    squared_error = mean_squared_error(observed, predicted)
    root_squared_error = np.sqrt(squared_error)
    absolute_error = mean_absolute_error(observed, predicted)
    return {
        'R2': coefficient,
        'RMSE': root_squared_error,
        'MAE': absolute_error,
    }


def build_preprocessing(missing):
    scaler = StandardScaler()
    if missing == 'median':
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        preprocessing = make_pipeline(imputer, scaler)
    else:
        preprocessing = make_pipeline(scaler)
    return preprocessing


def build_regressor(config, rounds):
    settings = config.copy()
    settings.pop('missing')
    settings['n_estimators'] = int(rounds)
    model = lgb.LGBMRegressor(
        **settings,
        random_state=SEED,
        n_jobs=MODEL_THREADS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    return model


def choose_tree_count(X, y, config):
    sample_indices = np.arange(len(y))
    fit_index, stop_index = train_test_split(
        sample_indices,
        test_size=0.15,
        random_state=SEED + 8,
    )
    fit_X = X[fit_index]
    fit_y = y[fit_index]
    stop_X = X[stop_index]
    stop_y = y[stop_index]

    preprocessing = build_preprocessing(config['missing'])
    transformed_fit_X = preprocessing.fit_transform(fit_X)
    transformed_stop_X = preprocessing.transform(stop_X)
    model = build_regressor(config, config['n_estimators'])

    fit_parameters = signature(model.fit).parameters
    if 'eval_X' in fit_parameters:
        validation_data = {'eval_X': transformed_stop_X, 'eval_y': stop_y}
    else:
        validation_data = {'eval_set': [(transformed_stop_X, stop_y)]}

    stopping_callback = lgb.early_stopping(
        EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    model.fit(
        transformed_fit_X,
        fit_y,
        **validation_data,
        callbacks=[stopping_callback],
    )
    best_iteration = int(model.best_iteration_)
    selected_rounds = max(1, best_iteration)
    return selected_rounds


def fit_model(X, y, config, rounds=None):
    if rounds is None:
        rounds = choose_tree_count(X, y, config)

    preprocessing = build_preprocessing(config['missing'])
    transformed_X = preprocessing.fit_transform(X)
    model = build_regressor(config, rounds)
    model.fit(transformed_X, y)

    bundle = {
        'model': model,
        'preprocessing': preprocessing,
        'rounds': int(rounds),
        'config': config.copy(),
    }
    return bundle


def predict_model(bundle, X):
    preprocessing = bundle['preprocessing']
    transformed_X = preprocessing.transform(X)
    booster = bundle['model'].booster_
    predicted = booster.predict(transformed_X, num_threads=MODEL_THREADS)
    return predicted


def evaluate_candidate(X, y, config, inner_splits):
    predictions = np.empty(len(y))
    selected_rounds = []

    for train_index, valid_index in inner_splits:
        train_X = X[train_index]
        train_y = y[train_index]
        valid_X = X[valid_index]

        bundle = fit_model(train_X, train_y, config)
        valid_predictions = predict_model(bundle, valid_X)
        predictions[valid_index] = valid_predictions
        selected_rounds.append(bundle['rounds'])

    pooled_mse = mean_squared_error(y, predictions)
    median_rounds = int(np.median(selected_rounds))
    return pooled_mse, median_rounds


def tune_model(X, y):
    inner_cv = KFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 1,
    )
    inner_splits = list(inner_cv.split(X))
    search_records = []

    for candidate_number, config in enumerate(candidates):
        pooled_mse, median_rounds = evaluate_candidate(X, y, config, inner_splits)
        record = {
            'Candidate': candidate_number,
            'Inner_MSE': pooled_mse,
            'Rounds': median_rounds,
        }
        search_records.append(record)

    search = pd.DataFrame(search_records)
    best_index = search['Inner_MSE'].idxmin()
    best_record = search.loc[best_index]
    selected_candidate = int(best_record['Candidate'])
    selected_rounds = int(best_record['Rounds'])
    selected_config = candidates[selected_candidate]

    bundle = fit_model(X, y, selected_config, rounds=selected_rounds)
    bundle['candidate'] = selected_candidate
    return bundle, search


# 4. Train and evaluate the outer-fold models

def select_features(arm):
    features = []
    for feature in FEATURES:
        if arm == 'Without NTL' and feature == 'NTL_UE':
            continue
        features.append(feature)
    return features


def calculate_heldout_shap(bundle, test_X, predicted, test_ids, key):
    preprocessing = bundle['preprocessing']
    transformed_X = preprocessing.transform(test_X)
    booster = bundle['model'].booster_
    contributions = booster.predict(
        transformed_X,
        pred_contrib=True,
        num_threads=MODEL_THREADS,
    )

    reconstructed_predictions = contributions.sum(axis=1)
    np.testing.assert_allclose(reconstructed_predictions, predicted, atol=1e-7, rtol=0)
    feature_contributions = contributions[:, :-1]
    expected_values = contributions[:, -1]

    shap_table = pd.DataFrame(feature_contributions, columns=bundle['features'])
    shap_table['Urban_lake_id'] = test_ids
    shap_table['Expected_value'] = expected_values
    shap_table = shap_table.assign(**key)
    return shap_table


def calculate_permutation_importance(bundle, test_X, test_y, baseline_r2, key):
    rows = []
    fold = key['Fold']
    partition = key['Partition']
    features = bundle['features']

    for repetition in range(PERMUTATIONS):
        permutation_seed = 91000 + fold * 100 + repetition
        if partition == 'New split':
            permutation_seed += 10000

        generator = np.random.default_rng(permutation_seed)
        order = generator.permutation(len(test_y))

        for column, feature in enumerate(features):
            permuted_X = test_X.copy()
            shuffled_column = permuted_X[order, column]
            permuted_X[:, column] = shuffled_column
            permuted_predictions = predict_model(bundle, permuted_X)
            permuted_r2 = r2_score(test_y, permuted_predictions)

            record = {
                **key,
                'Feature': feature,
                'Permutation': repetition,
                'R2_loss': baseline_r2 - permuted_r2,
            }
            rows.append(record)
    return rows


def run_outer_model(target, partition, fold, arm, train_index, test_index):
    features = select_features(arm)
    response_column = TARGETS[target]
    X = data[features].to_numpy(float)
    y = data[response_column].to_numpy(float)
    identifiers = data['Urban_lake_id'].to_numpy()

    train_X = X[train_index]
    train_y = y[train_index]
    test_X = X[test_index]
    test_y = y[test_index]
    test_ids = identifiers[test_index]

    bundle, search = tune_model(train_X, train_y)
    bundle['features'] = features.copy()
    bundle['target'] = target

    predicted = predict_model(bundle, test_X)
    scores = evaluate(test_y, predicted)
    key = {
        'Target': target,
        'Partition': partition,
        'Fold': fold,
        'Model': arm,
    }
    metrics = {
        **key,
        'N': len(test_index),
        **scores,
        'Candidate': bundle['candidate'],
        'Rounds': bundle['rounds'],
        'Missing': bundle['config']['missing'],
    }

    prediction_values = {
        'Urban_lake_id': test_ids,
        'Observed': test_y,
        'Predicted': predicted,
    }
    predictions = pd.DataFrame(prediction_values)
    predictions = predictions.assign(**key)
    search = search.assign(**key)

    shap_table = None
    permutation_rows = []
    if arm == 'With NTL':
        shap_table = calculate_heldout_shap(bundle, test_X, predicted, test_ids, key)
        permutation_rows = calculate_permutation_importance(
            bundle, test_X, test_y, scores['R2'], key
        )

    result = {
        'key': (target, partition, fold, arm),
        'bundle': bundle,
        'metrics': metrics,
        'predictions': predictions,
        'search': search,
        'shap': shap_table,
        'permutation': permutation_rows,
    }
    return result


jobs = []
for target in TARGETS:
    for partition, seed in PARTITIONS.items():
        outer_cv = KFold(OUTER_FOLDS, shuffle=True, random_state=seed)
        outer_splits = list(outer_cv.split(data))
        for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
            for arm in ['With NTL', 'Without NTL']:
                job = (target, partition, fold, arm, train_index, test_index)
                jobs.append(job)

outer_results = []
with ThreadPoolExecutor(max_workers=OUTER_WORKERS) as executor:
    futures = []
    for job in jobs:
        future = executor.submit(run_outer_model, *job)
        futures.append(future)

    for completed, future in enumerate(as_completed(futures), start=1):
        result = future.result()
        outer_results.append(result)
        row = result['metrics']
        print(f"{completed:02d}/{len(jobs)} | {row['Target']} | {row['Partition']} | "
              f"fold {row['Fold']} | {row['Model']} | R² = {row['R2']:.4f}", flush=True)

outer_results.sort(key=lambda result: result['key'])
metric_records = []
prediction_tables = []
search_tables = []
shap_tables = []
permutation_records = []
outer_models = {}

for result in outer_results:
    metric_records.append(result['metrics'])
    prediction_tables.append(result['predictions'])
    search_tables.append(result['search'])
    if result['shap'] is not None:
        shap_tables.append(result['shap'])
    permutation_records.extend(result['permutation'])
    outer_models[result['key']] = result['bundle']

fold_metrics = pd.DataFrame(metric_records)
oof_predictions = pd.concat(prediction_tables, ignore_index=True)
inner_search = pd.concat(search_tables, ignore_index=True)
heldout_shap = pd.concat(shap_tables, ignore_index=True)
permutation_results = pd.DataFrame(permutation_records)


# 5. Summarize validation performance and selected parameters

pooled_rows = []
group_columns = ['Target', 'Partition', 'Model']
prediction_groups = oof_predictions.groupby(group_columns)

for (target, partition, arm), group in prediction_groups:
    observed_values = group['Observed']
    predicted_values = group['Predicted']
    pooled_scores = evaluate(observed_values, predicted_values)
    summary_row = {
        'Target': target,
        'Partition': partition,
        'Model': arm,
        'N': len(group),
        **pooled_scores,
    }
    pooled_rows.append(summary_row)

pooled_metrics = pd.DataFrame(pooled_rows)
display(pooled_metrics.round(5))

selection_columns = [
    'Target', 'Partition', 'Fold', 'Model', 'Candidate', 'Rounds', 'Missing',
]
selection_table = fold_metrics[selection_columns]
parameter_details = candidate_table.drop(columns='missing')
selected_parameters = selection_table.merge(
    parameter_details,
    on='Candidate',
    how='left',
)
parameter_display_columns = [column for column in selection_columns if column != 'Missing'] + [
    'learning_rate', 'num_leaves', 'min_child_samples',
]
display(selected_parameters[parameter_display_columns])


# 6. SHAP rank stability

shap_rows = []
for (target, partition, fold), group in heldout_shap.groupby(['Target', 'Partition', 'Fold']):
    importance = group[FEATURES].abs().mean()
    ranks = importance.rank(ascending=False, method='average')
    for feature in FEATURES:
        shap_rows.append({
            'Target': target, 'Partition': partition, 'Fold': fold, 'Feature': feature,
            'N': len(group), 'Mean_abs_SHAP': importance[feature], 'Rank': ranks[feature],
            'Relative_percent': 100 * importance[feature] / importance.sum(),
        })
shap_fold = pd.DataFrame(shap_rows)
shap_summary_rows = []
for (target, feature), group in shap_fold.groupby(['Target', 'Feature']):
    shap_summary_rows.append({
        'Target': target, 'Feature': feature,
        'Pooled_mean_abs_SHAP': np.average(group['Mean_abs_SHAP'], weights=group['N']),
        'Min_rank': group['Rank'].min(), 'Max_rank': group['Rank'].max(),
        'Top_three_fraction': (group['Rank'] <= 3).mean(),
    })
shap_summary = pd.DataFrame(shap_summary_rows)
shap_summary['Pooled_rank'] = shap_summary.groupby('Target')['Pooled_mean_abs_SHAP'].rank(ascending=False)

labels = {
    'Temperature_UE': 'Temperature', 'NTL_UE': 'NTL', 'VPD_UE': 'VPD',
    'Precipitation_UE': 'Precipitation', 'Runoff_UE': 'Runoff', 'Wind_speed_UE': 'Wind speed',
    'Radiation_UE': 'Radiation', 'Soil_moisture_UE': 'Soil moisture', 'LAI_UE': 'LAI',
}
colors = {'Original split': '#35759A', 'New split': '#BD7946'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 130,
})

fig_rank, axes = plt.subplots(1, 2, figsize=(12, 6))
fig_rank.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.91, wspace=0.58)
for ax, target, letter in zip(axes, TARGETS, ['a', 'b']):
    order = shap_summary.loc[shap_summary['Target'] == target].sort_values('Pooled_rank')['Feature']
    for row, feature in enumerate(order):
        group = shap_fold.loc[(shap_fold['Target'] == target) & (shap_fold['Feature'] == feature)]
        ax.plot([group['Rank'].min(), group['Rank'].max()], [row, row], color='#B8B8B8', lw=1.4, zorder=0)
        for partition, marker, offset in [('Original split', 'o', -0.09), ('New split', '^', 0.09)]:
            points = group.loc[group['Partition'] == partition].sort_values('Fold')
            ax.scatter(points['Rank'], row + offset + np.linspace(-0.035, 0.035, len(points)),
                       s=27, color=colors[partition], marker=marker)
    ax.set(yticks=range(9), yticklabels=[labels[f] for f in order], xticks=range(1, 10),
           xlim=(0.6, 9.4), xlabel='Held-out SHAP rank')
    ax.invert_yaxis()
    ax.set_title(target, loc='left', pad=15)
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
handles = [Line2D([], [], color=colors[p], marker=m, linestyle='', label=p)
           for p, m in [('Original split', 'o'), ('New split', '^')]]
fig_rank.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.045), ncol=2)
plt.show()


# 7. Marginal permutation importance

permutation_fold = permutation_results.groupby(
    ['Target', 'Partition', 'Fold', 'Feature'], as_index=False
).agg(Mean_R2_loss=('R2_loss', 'mean'))
permutation_summary = permutation_fold.groupby(['Target', 'Feature'], as_index=False).agg(
    Mean_R2_loss=('Mean_R2_loss', 'mean'), Min_R2_loss=('Mean_R2_loss', 'min'),
    Max_R2_loss=('Mean_R2_loss', 'max'),
)

fig_permutation, axes = plt.subplots(1, 2, figsize=(12, 6))
fig_permutation.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.91, wspace=0.58)
for ax, target, letter in zip(axes, TARGETS, ['a', 'b']):
    panel = permutation_summary.loc[permutation_summary['Target'] == target].sort_values('Mean_R2_loss', ascending=False)
    for row_number, row in enumerate(panel.itertuples()):
        color = '#BD7946' if row.Feature == 'NTL_UE' else '#35759A'
        ax.plot([row.Min_R2_loss, row.Max_R2_loss], [row_number, row_number], color=color, lw=1.4)
        ax.scatter(row.Mean_R2_loss, row_number, color=color, s=46, zorder=3)
        points = permutation_fold.loc[(permutation_fold['Target'] == target) & (permutation_fold['Feature'] == row.Feature)]
        ax.scatter(points['Mean_R2_loss'], row_number + np.linspace(-0.12, 0.12, len(points)),
                   s=16, color=color, alpha=0.35)
    ax.axvline(0, color='#777777', lw=0.8)
    ax.set(yticks=range(9), yticklabels=panel['Feature'].map(labels),
           xlabel='R² loss after marginal permutation')
    ax.invert_yaxis()
    ax.set_title(target, loc='left', pad=15)
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
plt.show()


# 8. Independently tuned models with and without NTL

comparison_keys = ['Target', 'Partition']
paired_folds = fold_metrics.loc[fold_metrics['Model'] == 'With NTL'].merge(
    fold_metrics.loc[fold_metrics['Model'] == 'Without NTL'],
    on=comparison_keys + ['Fold'], suffixes=('_with', '_without'),
)
paired_pooled = pooled_metrics.loc[pooled_metrics['Model'] == 'With NTL'].merge(
    pooled_metrics.loc[pooled_metrics['Model'] == 'Without NTL'],
    on=comparison_keys, suffixes=('_with', '_without'),
)
paired_pooled['R2_loss_on_NTL_removal'] = paired_pooled['R2_with'] - paired_pooled['R2_without']
display(paired_pooled[comparison_keys + ['R2_with', 'R2_without', 'R2_loss_on_NTL_removal']].round(5))

fig_ntl, axes = plt.subplots(1, 2, figsize=(12, 5.4))
fig_ntl.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.89, wspace=0.35)
for ax, target, letter in zip(axes, TARGETS, ['a', 'b']):
    for partition in PARTITIONS:
        folds = paired_folds.loc[(paired_folds['Target'] == target) & (paired_folds['Partition'] == partition)]
        for row in folds.itertuples():
            ax.plot([0, 1], [row.R2_with, row.R2_without], color=colors[partition],
                    alpha=0.22, lw=0.9, marker='o', ms=3)
        row = paired_pooled.loc[(paired_pooled['Target'] == target) & (paired_pooled['Partition'] == partition)].iloc[0]
        ax.plot([0, 1], [row['R2_with'], row['R2_without']], color=colors[partition],
                lw=2.5, marker='o', ms=6)
    ax.set(xticks=[0, 1], xticklabels=['With NTL', 'Without NTL'],
           xlim=(-0.15, 1.15), ylabel='Held-out R²')
    ax.set_title(target, loc='left', pad=15)
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')
handles = [Line2D([], [], color=colors[p], lw=2.5, label=p) for p in PARTITIONS]
fig_ntl.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.05), ncol=2)
plt.show()




