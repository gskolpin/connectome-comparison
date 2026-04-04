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

# %load_ext autoreload
# %autoreload 2

# %%
#here need to read in the results again and channge things below as I go... 
model_names = ['ridgeCV', 'lassoCV', 'LinearRegression', 'PCLasso', 'cpm']
data_names = ['lasso', 'corr', 'net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr']
stats = ['r', 'full_r2', 'MAE']
descriptives = {model:{data:{stat:[] for stat in stats} for data in data_names} for model in model_names}

predictions_filepath = '/gscratch/scrubbed/gkolpin/age_predictions'
batch = 'batch_test'

for file in glob.glob(op.join(predictions_filepath,
                                 batch,
                                 '*results.pkl')):
            with open(file, 'rb') as i:
                loaded_data = pkl.load(i)
                for model in model_names:
                    for data in data_names:
                        for stat in stats:
                            descriptives[model][data][stat].append(loaded_data[model][data][stat])

index = pd.MultiIndex.from_product([model_names, data_names], names=('Model', 'Data'))
columns = ['r', 'r_CI', 'r_SD', 'full_r2', 'r2_CI', 'r2_SD', 'MAE', 'MAE_CI', 'MAE_SD']
results = pd.DataFrame(index=index, columns=columns)
stat_metrics = [('r', 'r_CI', 'r_SD'), ('full_r2', 'r2_CI', 'r2_SD'), ('MAE', 'MAE_CI', 'MAE_SD')]
for model in model_names:
    for data in data_names:
        for stat in stat_metrics:
            results.loc[(model, data), stat[0]] = statistics.mean(descriptives[model][data][stat[0]])
            results.loc[(model, data), stat[1]] = (np.percentile(descriptives[model][data][stat[0]], 97.5) - np.percentile(descriptives[model][data][stat[0]], 2.5)) / 2
            results.loc[(model, data), stat[2]] = statistics.stdev(descriptives[model][data][stat[0]])

# %%
#Define data at the start because all of them is just too much
data = 'lasso'
model_names = ['ridgeCV', 'lassoCV', 'LinearRegression', 'PCLasso', 'cpm']

stats = ['r', 'full_r2', 'MAE']
fig, axs = plt.subplots(5, 3, sharey=True, figsize=(12, 16))

for row, model in zip(axs, model_names):
    for col, stat in zip(row, stats):
        col.hist(descriptives[model][data][stat])
        col.set_title(f'{model} {stat}')
plt.tight_layout()

# %%
results.head()

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
print('lasso: r2 SD:', lasso_results['r2_SD'], 'r SD:', lasso_results['r_SD'], 'MAE SD:', lasso_results['MAE_SD'],
    '\ncorrelation: r2 SD', corr_results['r2_SD'], 'r SD:', corr_results['r_SD'], 'MAE SD:', corr_results['MAE_SD'],
    '\nlasso 7 networks: r2 SD', net7_lasso_results['r2_SD'], 'r SD:', net7_lasso_results['r_SD'], 'MAE SD:', net7_lasso_results['MAE_SD'],
    '\nnet17 lasso, r2:', net17_lasso_results['r2_SD'], 'r SD:', net17_lasso_results['r_SD'], 'MAE SD:', net17_lasso_results['MAE_SD'],
    '\ncorrelation 7 networks: r2 SD', net7_corr_results['r2_SD'], 'r SD:', net7_corr_results['r_SD'], 'MAE SD:', net7_corr_results['MAE_SD'],
    '\ncorrelation 17 networks: r2 SD', net17_corr_results['r2_SD'], 'r SD:', net17_corr_results['r_SD'], 'MAE SD:', net17_corr_results['MAE_SD'])

# %%
#EVERYTHING BELOW DOESN'T WORK because I now am lacking residuals and actual data points. Could try and save a full dataset model somewhere else...

# %%
plt.scatter(lasso_results['k_fold_age'], lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(lasso_results['k_fold_age'], lasso_results['predictions'], 1)
plt.plot(pca_lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')

plt.title('PCA lasso age prediction')
plt.show()
print('lasso:', lasso_results['fold 1 stats'], 'full r2:', lasso_results['full_r2'])

# %%
#plotting residuals
plt.scatter(corr_results['k_fold_age'], corr_results['residuals'])

# %%
f, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 16))
axs[0, 0].scatter(corr_results['k_fold_age'], corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(corr_results['k_fold_age'], corr_results['predictions'], 1)
axs[0, 0].plot(corr_results['k_fold_age'], corr_a + corr_b * corr_data['age'], color='red')
corr00 = np.corrcoef(corr_results['k_fold_age'], corr_results['predictions'])[1, 0] ** 2
axs[0, 0].annotate(f'$R^2$={corr00:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[0, 1].scatter(lasso_results['k_fold_age'], lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(lasso_results['k_fold_age'], lasso_results['predictions'], 1)
axs[0, 1].plot(lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')
corr01 = np.corrcoef(lasso_results['k_fold_age'], lasso_results['predictions'])[1, 0] ** 2
axs[0, 1].annotate(f'$R^2$={corr01:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[1, 0].scatter(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], 1)
axs[1, 0].plot(net7_corr_results['k_fold_age'], n7_a + n7_b * net7_corr_data['age'], color='red')
corr10 = np.corrcoef(net7_corr_results['k_fold_age'], net7_corr_results['predictions'])[1, 0] ** 2
axs[1, 0].annotate(f'$R^2$={corr10:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[1, 1].scatter(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], 1)
axs[1, 1].plot(net7_lasso_results['k_fold_age'], n7_a + n7_b * net7_lasso_data['age'], color='red')
corr11 = np.corrcoef(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'])[1, 0] ** 2
axs[1, 1].annotate(f'$R^2$={corr11:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[2, 0].scatter(net17_corr_results['k_fold_age'],net17_corr_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_corr_results['k_fold_age'], net17_corr_results['predictions'], 1)
axs[2, 0].plot(net17_corr_results['k_fold_age'], n17_a + n17_b * net17_corr_data['age'], color='red')
corr20 = np.corrcoef(net17_corr_results['k_fold_age'], net17_corr_results['predictions'])[1, 0] ** 2
axs[2, 0].annotate(f'$R^2$={corr20:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[2, 1].scatter(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], 1)
axs[2, 1].plot(net17_lasso_results['k_fold_age'], n17_a + n17_b * net17_lasso_data['age'], color='red')
corr21 = np.corrcoef(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'])[1, 0] ** 2
axs[2, 1].annotate(f'$R^2$={corr21:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

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
