# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: FC
#     language: python
#     name: fc
# ---

# %%
import glob
import os
import os.path as op

import pandas as pd
import numpy as np
import math
import statistics
import scipy
from copy import deepcopy
import pickle as pkl
import pprint
import json

import nilearn
import nilearn.datasets
from nilearn.plotting import plot_roi, show

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import r_regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.utils import resample
from scipy.stats import sem

from fracridge import fracridge, FracRidgeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from functions import get_connectomes
from functions import cpm
from functions import fit_model
import tol_colors as tc

# %load_ext autoreload
# %autoreload 2

# %%
atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_filename = atlas.maps
atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas17_filename = atlas.maps

full_labels = atlas.labels[1:]
labels_7 = [label.split("_")[2] for label in full_labels]

full_labels_17 = atlas_17.labels[1:]
labels_17 = [label.split("_")[2] for label in full_labels_17]

# %%
#defining stuff about the data to make it actually run.
batch = 'batch_task'
networks = True #does the data have network levels?
cpm = True #Is cpm part of it? 
pca = False #is pclasso part of it?
weights = False #do I have model weights (cpm and PClasso don't have weights)

# %%
#here need to read in the results again and channge things below as I go... 
data_names = ['lasso', 'corr']
full_connectome_len = 10000
connectome_column_labels = [[f'edge_{num}' for num in range(full_connectome_len)], 
                         [f'edge_{num}' for num in range(full_connectome_len)]] 
if networks:
    net7_length = 7**2
    net17_length = 17**2
    data_names.extend(['net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr'])
    connectome_column_labels.extend([labels_7, 
                              labels_7, 
                              labels_17, 
                              labels_17])
model_names = ['ridgeCV', 'lassoCV', 'LinearRegression']
if cpm:
    model_names.append('cpm')
if pca:
    model_names.append('PCLasso')
stats = ['r', 'full_r2', 'MAE']
descriptives = {model:{data:{stat:[] for stat in stats} for data in data_names} for model in model_names}
model_coefs = {model:{data:[] for data in data_names} for model in model_names}

predictions_filepath = '/gscratch/scrubbed/gkolpin/age_predictions'

# %%
for file in glob.glob(op.join(predictions_filepath,
                                 batch,
                                 '*results.pkl')):
    with open(file, 'rb') as i:
        loaded_data = pkl.load(i)
        for model in model_names:
            for data in data_names:
                for stat in stats:
                    descriptives[model][data][stat].append(loaded_data[model][data][stat])
                if weights:
                    model_coefs[model][data].append(list(loaded_data[model][data]['avg_coefs']))

# %%
plt.hist(descriptives['LinearRegression']['corr']['full_r2'], color='black', alpha=.6)
ptop = np.percentile(descriptives['LinearRegression']['corr']['full_r2'], 97.5)
pbot = np.percentile(descriptives['LinearRegression']['corr']['full_r2'], 2.5)
plt.axvline(ptop, color='darkblue', linestyle='dashed', linewidth=2, label=f'97.5th Percentile: {ptop:.2f}')
plt.axvline(pbot, color='darkblue', linestyle='dashed', linewidth=2, label=f'3.5th Percentile: {pbot:.2f}')
plt.axis('off')
#plt.savefig('dist')

# %%
fig, axs = plt.subplots(len(model_names), 3, sharey=True, sharex='col', figsize=(12, 16))

for row, model in zip(axs, model_names):
    for col, stat in zip(row, stats):
        col.hist(descriptives[model]['corr'][stat], color='red', alpha=.5, label='corr')
        col.hist(descriptives[model]['lasso'][stat], color='blue', alpha=.5, label='lasso')
        col.legend()
        col.set_title(f'{model} {stat}')
        col.tick_params(labelbottom=True)
plt.tight_layout()
#plt.savefig('p_prediciton_results')

# %%
index = pd.MultiIndex.from_product([model_names, data_names], names=('Model', 'Data'))
columns = ['r', 'r_CI', 'r_SD', 'full_r2', 'r2_CI', 'r2_SD', 'MAE', 'MAE_CI', 'MAE_SD']
results = pd.DataFrame(index=index, columns=columns)
stat_metrics = [('r', 'r_CI', 'r_SD'), ('full_r2', 'r2_CI', 'r2_SD'), ('MAE', 'MAE_CI', 'MAE_SD')]
for model in model_names:
    for data in data_names:
        for stat in stat_metrics:
            results.loc[(model, data), stat[0]] = statistics.mean(descriptives[model][data][stat[0]])
            results.loc[(model, data), stat[1]] = np.percentile(descriptives[model][data][stat[0]], [2.5, 97.5])
            results.loc[(model, data), stat[2]] = statistics.stdev(descriptives[model][data][stat[0]])

# %%
#for everything below Im defining a certain model and comparing the different connectomes when predicting age from this model 
model = 'ridgeCV'
lasso_results = results.loc[(model, 'lasso')]
corr_results = results.loc[(model, 'corr')]
net7_lasso_results = results.loc[(model, 'net7_lasso')]
net7_corr_results = results.loc[(model, 'net7_corr')]
net17_lasso_results = results.loc[(model, 'net17_lasso')]
net17_corr_results = results.loc[(model, 'net17_corr')]

# %%
#well try a violin plot
#okay so for this 
model='ridgeCV'
medium = tc.colorsets['medium_contrast']

stats = ['MAE', 'full_r2', 'r']
CIs = ['r2_CI', 'MAE_CI']

#for col, stat, ci in zip(axs, stats, CIs):
lasso_data = pd.DataFrame()
for stat in stats:
    stat_data = pd.DataFrame(descriptives[model]['lasso'][stat], columns=['Value'])
    stat_data['stat'] = stat
    stat_data['model'] = 'Lasso'
    lasso_data = pd.concat([stat_data, lasso_data], axis=0, join='outer')

corr_data = pd.DataFrame()
for stat in stats:
    stat_data = pd.DataFrame(descriptives[model]['corr'][stat], columns=['Value'])
    stat_data['stat'] = stat
    stat_data['model'] = 'Pearson'
    corr_data = pd.concat([stat_data, corr_data], axis=0, join='outer')

#corr_data.rename(columns={column: f'corr {column}' for column in corr_data.columns}, inplace=True)
data = pd.concat([lasso_data, corr_data], axis=0, join='outer')

sns.violinplot(data, x='stat', 
               y='Value', 
               ]hue='model', 
               split=True, 
               fill=True, 
               inner='quart', 
               palette={'Pearson': medium.light_blue, 'Lasso': medium.dark_yellow}, 
               alpha=.5
              )

# %%
f, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 6))
stats = ['full_r2', 'MAE']
CIs = ['r2_CI', 'MAE_CI']

for col, stat, ci in zip(axs, stats, CIs):
    col.hist(descriptives['ridgeCV']['corr'][stat], color=medium.light_blue, alpha=0.5, label='Pearson')
    col.axvline(corr_results[ci][0], color=medium.light_blue, alpha=.3, linestyle='dashed', linewidth=2)
    col.axvline(corr_results[ci][1], color=medium.light_blue, alpha=.3, linestyle='dashed', linewidth=2)
    col.axvline(corr_results[stat], color=medium.light_blue, alpha=.6, linewidth=2)
    
    col.hist(descriptives['ridgeCV']['lasso'][stat], color=medium.dark_yellow, alpha=0.5, label='Lasso')
    col.axvline(lasso_results[ci][0], color=medium.dark_yellow, alpha=.3, linestyle='dashed', linewidth=2)
    col.axvline(lasso_results[ci][1], color=medium.dark_yellow, alpha=.3, linestyle='dashed', linewidth=2)
    col.axvline(lasso_results[stat], color=medium.dark_yellow, alpha=.6, linewidth=2)
    
    col.legend()

axs[0].set_title('Ridge Age Prediciton R2')
axs[1].set_title('Ridge Age Prediciton MAE')
axs[0].set_ylabel('count')
plt.tight_layout()
#plt.savefig('ridge age dists')

# %%
print('lasso: r2:', lasso_results['full_r2'], 'r:', lasso_results['r'], 'MAE:', lasso_results['MAE'],
    '\ncorrelation:', corr_results['full_r2'], 'r:', corr_results['r'], 'MAE:', corr_results['MAE'],
    '\nlasso 7 networks:', net7_lasso_results['full_r2'], 'r:', net7_lasso_results['r'], 'MAE:', net7_lasso_results['MAE'],
    '\nnet17 lasso, r2:', net17_lasso_results['full_r2'], 'r:', net17_lasso_results['r'], 'MAE:', net17_lasso_results['MAE'],
    '\ncorrelation 7 networks:', net7_corr_results['full_r2'], 'r:', net7_corr_results['r'], 'MAE:', net7_corr_results['MAE'],
    '\ncorrelation 17 networks:', net17_corr_results['full_r2'], 'r:', net17_corr_results['r'], 'MAE:', net17_corr_results['MAE'])

# %%
print('lasso: r2 CI:', lasso_results['r2_CI'], 'r CI:', lasso_results['r_CI'], 'MAE CI:', lasso_results['MAE_CI'],
    '\ncorrelation: r2 CI', corr_results['r2_CI'], 'r CI:', corr_results['r_CI'], 'MAE CI:', corr_results['MAE_CI'],
    '\nlasso 7 networks: r2 CI', net7_lasso_results['r2_CI'], 'r CI:', net7_lasso_results['r_CI'], 'MAE CI:', net7_lasso_results['MAE_CI'],
    '\nnet17 lasso, r2:', net17_lasso_results['r2_CI'], 'r CI:', net17_lasso_results['r_CI'], 'MAE CI:', net17_lasso_results['MAE_CI'],
    '\ncorrelation 7 networks: r2 CI', net7_corr_results['r2_CI'], 'r CI:', net7_corr_results['r_CI'], 'MAE CI:', net7_corr_results['MAE_CI'],
    '\ncorrelation 17 networks: r2 CI', net17_corr_results['r2_CI'], 'r CI:', net17_corr_results['r_CI'], 'MAE CI:', net17_corr_results['MAE_CI'])

# %%
print('lasso: r2 SD:', lasso_results['r2_SD']*2, 'r SD:', lasso_results['r_SD']*2, 'MAE SD:', lasso_results['MAE_SD']*2,
    '\ncorrelation: r2 SD', corr_results['r2_SD']*2, 'r SD:', corr_results['r_SD']*2, 'MAE SD:', corr_results['MAE_SD']*2,
    '\nlasso 7 networks: r2 SD', net7_lasso_results['r2_SD']*2, 'r SD:', net7_lasso_results['r_SD']*2, 'MAE SD:', net7_lasso_results['MAE_SD']*2,
    '\nnet17 lasso, r2:', net17_lasso_results['r2_SD']*2, 'r SD:', net17_lasso_results['r_SD']*2, 'MAE SD:', net17_lasso_results['MAE_SD']*2,
    '\ncorrelation 7 networks: r2 SD', net7_corr_results['r2_SD']*2, 'r SD:', net7_corr_results['r_SD']*2, 'MAE SD:', net7_corr_results['MAE_SD']*2,
    '\ncorrelation 17 networks: r2 SD', net17_corr_results['r2_SD']*2, 'r SD:', net17_corr_results['r_SD']*2, 'MAE SD:', net17_corr_results['MAE_SD']*2)

# %%
#specifically for p prediction stuff
model = 'cpm'
datas = ['corr', 'lasso']

f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 5))
for col, data in zip(axs, datas):
    col.scatter(loaded_data[model][data]['k_fold_age'], loaded_data[model][data]['predictions'])
    col.axhline(0, linestyle=':', color='black')
    col.axvline(0, linestyle=':', color='black')
#plt.xlim(-2, 2)
#plt.ylim(-2, 2)
axs[0].set_xlabel('True')
axs[1].set_xlabel('True')
axs[0].set_ylabel('Predicted')
axs[0].set_title('Predicted from Corr connectomes')
axs[1].set_title('Predicted from Lasso connectomes')

# %%
model = 'ridgeCV'
datas = ['corr', 'lasso']

f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 5))
for col, data in zip(axs, datas):
    col.scatter(loaded_data[model][data]['k_fold_age'], loaded_data[model][data]['predictions'])
    col.axhline(0, linestyle=':', color='black')
    col.axvline(0, linestyle=':', color='black')
#plt.xlim(-2, 2)
#plt.ylim(-2, 2)
axs[0].set_xlabel('True')
axs[1].set_xlabel('True')
axs[0].set_ylabel('Predicted')
axs[0].set_title('Predicted from Corr connectomes')
axs[1].set_title('Predicted from Lasso connectomes')

# %%
if weights:
    for model in model_names:
                for data in data_names:
                    coef_array = np.array(model_coefs[model][data])
                    coef_array = coef_array.mean(axis=0)
                    model_coefs[model][data] = coef_array

# %%
if weights:
    def row_limits(*arrays):
        stacked = np.concatenate([a.ravel() for a in arrays])
        m = np.max(np.abs(stacked))
        return -m, m
    
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    
    mat1 = np.reshape(model_coefs[model]['lasso'], (100, 100))
    mat2 = np.reshape(model_coefs[model]['corr'], (100, 100))
    vmin, vmax = row_limits(mat1, mat2)
    sns.heatmap(mat1, ax=axs[0, 0], cmap='bwr', vmin=vmin, vmax=vmax)
    sns.heatmap(mat2, ax=axs[0, 1], cmap='bwr', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("lasso")
    axs[0, 1].set_title("corr")
    
    mat3 = np.reshape(model_coefs[model]['net7_lasso'], (7, 7))
    mat4 = np.reshape(model_coefs[model]['net7_corr'], (7, 7))
    vmin, vmax = row_limits(mat3, mat4)
    sns.heatmap(mat3, ax=axs[1, 0], cmap='bwr', vmin=vmin, vmax=vmax)
    sns.heatmap(mat4, ax=axs[1, 1], cmap='bwr', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("net7_lasso")
    axs[1, 1].set_title("net7_corr")
    
    mat5 = np.reshape(model_coefs[model]['net17_lasso'], (17, 17))
    mat6 = np.reshape(model_coefs[model]['net17_corr'], (17, 17))
    vmin, vmax = row_limits(mat5, mat6)
    sns.heatmap(mat5, ax=axs[2, 0], cmap='bwr', vmin=vmin, vmax=vmax)
    sns.heatmap(mat6, ax=axs[2, 1], cmap='bwr', vmin=vmin, vmax=vmax)
    axs[2, 0].set_title("net17_lasso")
    axs[2, 1].set_title("net17_corr")
    
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# %%
#here can put the full batch
full_filepath = '/gscratch/scrubbed/gkolpin/age_predictions/batch_full_709/0_results.pkl'
full_data_names = ['lasso', 'corr', 'net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr']
full_model_names = ['ridgeCV', 'lassoCV', 'LinearRegression']
full_results = {model:{data:[] for data in full_data_names} for model in full_model_names}
with open(full_filepath, 'rb') as i:
    loaded_data = pkl.load(i)
    for model in full_model_names:
        for data in full_data_names:
            full_results[model][data] = loaded_data[model][data]
            full_results[model][data]['k_fold_age'] = np.array(full_results[model][data]['k_fold_age']).astype(float)

# %%
#Now can re-write the results into the full model, and everything else should run...
#But again, need to specify the model:
model = 'ridgeCV'
lasso_results = full_results[model]['lasso']
corr_results = full_results[model]['corr']
net7_lasso_results = full_results[model]['net7_lasso']
net7_corr_results = full_results[model]['net7_corr']
net17_lasso_results = full_results[model]['net17_lasso']
net17_corr_results = full_results[model]['net17_corr']

# %%
#plotting residuals... uh oh, that doesn't look very homoskedastic... well...
plt.scatter(corr_results['k_fold_age'], corr_results['residuals'])

# %%
f, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(12, 6))
all_results = [[corr_results, lasso_results]]
i = 1
for connectome in all_results:
    for col, res in zip(axs, connectome):
        x = res['k_fold_age']
        y = res['predictions']
        col.scatter(x, y, alpha=0.5, color='grey')
        sns.regplot(res, x=x, y=y, ci=99, ax=col, scatter=False, color=['red' if i%2 == 0 else 'blue'][0])

        col.plot([8, 23], [8, 23], linestyle=':', color='black')
        col.set_xlim(8, 23)
        col.set_ylim(8, 23)
        col.set_xlabel('Age')
        i += 1

axs[0].set_title('Pearson Connectome Predictions')
axs[1].set_title('Lasso Connectome Predictions')
axs[0].set_ylabel('Predicted Age')
plt.tight_layout()
#plt.savefig('predicted v actuall ridge')

# %%
f, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 16))
all_results = [[corr_results, lasso_results], [net7_corr_results, net7_lasso_results], [net17_corr_results, net17_lasso_results]]
i = 1
for row, connectome in zip(axs, all_results):
    for col, res in zip(row, connectome):
        x = res['k_fold_age']
        y = res['predictions']
        col.scatter(x, y, alpha=0.5, color='grey')
        sns.regplot(res, x=x, y=y, ax=col, scatter=False, color=['red' if i%2 == 0 else 'blue'][0])

        col.plot([8, 23], [8, 23], linestyle=':', color='black')
        col.set_xlim(8, 23)
        col.set_ylim(8, 23)
        i += 1

axs[0, 0].set_title('Correlation Full')
axs[0, 1].set_title('Lasso Full')
axs[1, 0].set_title('Correlation 7 Network')
axs[1, 1].set_title('Lasso 7 Network')
axs[2, 0].set_title('Correlation 17 Network')
axs[2, 1].set_title('Lasso 17 Network')
plt.show()

# %%
'''
#all set on its own, just for the figure in the writeup
model_ridge = RidgeCV()
folds = 5
lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], model_ridge, folds=folds)

plt.figure(figsize=(10, 5)) 
plt.scatter(lasso_results['k_fold_age'], lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(lasso_results['k_fold_age'], lasso_results['predictions'], 1)
plt.plot(lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')
plt.xlabel('age (years)')
plt.ylabel('predicted age (years)', rotation=90)
plt.xlim(7, 24)
plt.ylim(7, 24)
plt.title('Age predicitons from LASSO connectomes / Ridge prediction')

corr = np.corrcoef(lasso_results['k_fold_age'], lasso_results['predictions'])[1, 0] ** 2
plt.annotate(f'$R^2$={corr:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
'''

# %%
'''
#all set on its own, just for the figure in the writeup
model_lasso = LassoCV()
folds = 5
corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], model_lasso, folds=folds)

plt.figure(figsize=(10, 5)) 
plt.scatter(corr_results['k_fold_age'], corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(corr_results['k_fold_age'], corr_results['predictions'], 1)
plt.plot(corr_results['k_fold_age'], corr_a + corr_b * corr_data['age'], color='red')
plt.xlabel('age (years)')
plt.ylabel('predicted age (years)', rotation=90)
plt.xlim(7, 24)
plt.ylim(7, 24)
plt.title('Age predicitons from Pearson connectomes / LASSO prediction')

corr = np.corrcoef(corr_results['k_fold_age'], corr_results['predictions'])[1, 0] ** 2
plt.annotate(f'$R^2$={corr:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
'''

# %%
#okay so stratified by age
lasso_data['residuals'] = lasso_results['residuals']
data = lasso_data
age_brackets = [8, 10, 12, 14, 16, 18, 20, 22]
for lower in age_brackets:
    upper = lower + 2
    filtered_data = data[data['age'].between(lower, upper)]
    '''
    model = RidgeCV()
    upper = lower + 2
    filtered_data = data[data["age"].between(lower, upper)]
    filtered_results = fit_model(filtered_data[[f'edge_{num}' for num in range(10000)]], filtered_data['age'], model, folds=5, component='PCA')
    print(f'ages {lower} to {upper}', filtered_results['fold 1 stats'], 'full r2:', filtered_results['full_r2'])
    '''
    print(filtered_data['residuals'].mean())

# %%
#TO DO: get the nilearn correlaiton data visualizations working??? (the cai figure 3) (this is definetly low on priorities rn though)

# %%
#TO DO: correlate residuals with p values??? at least plot them... if I see outliers like their 10% i could remove those
#really funny, completely forgot this is not the stat p value, but the p score, and that took me a sec.

# %%
#below from the pprediction file, need to see if its worth keeping

# %%
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(cpm_corr_results['k_fold_y'], cpm_corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(cpm_corr_results['k_fold_y'], cpm_corr_results['predictions'], 1)
ax1.plot(cpm_corr_results['k_fold_y'], corr_a + corr_b * corr_data[p], color='red')

ax2.scatter(cpm_lasso_results['k_fold_y'], cpm_lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(cpm_lasso_results['k_fold_y'], cpm_lasso_results['predictions'], 1)
ax2.plot(cpm_lasso_results['k_fold_y'], lasso_a + lasso_b * lasso_data[p], color='red')

ax1.set_title('CPM p correlation')
ax2.set_title('CPM p lasso')
plt.show()

# %%
