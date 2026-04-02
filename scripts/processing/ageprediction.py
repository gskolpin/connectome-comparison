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
# Ideally everything hardcoded here, so can just switch these
#filepath to the csv connectivity files themselves
con_filepath = '/gscratch/scrubbed/gkolpin/xcpd_output/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd'
#and model
con_model = 'lassoBIC_blocks'
#filepath for the predictor or just all the phenotype data (need to make a space for those in scrubbed)
p_filepath = '/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv'
scaler = StandardScaler() #making it normal so things work with it
sublist_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/rand709_sub_list.txt'
random_state = 10

# %%
sublist = []
with open(sublist_filepath, 'r') as file:
    sublist = file.read().splitlines()
print(len(sublist))

# %%
#lets just first get the atlas image
atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_filename = atlas.maps
#plot_roi(atlas_filename, title="Schaefer_2018 atlas", view_type="contours")

atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas17_filename = atlas.maps

# %%
#now see if we can get the edge network labels
full_labels = atlas.labels[1:]
labels_7 = [label.split("_")[2] for label in full_labels]

full_labels_17 = atlas_17.labels[1:]
labels_17 = [label.split("_")[2] for label in full_labels_17]

# %%
task_connectomes, task_tp = get_connectomes(con_model='lassoBIC_task')

# %%
net7_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_7)[0]
net17_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_17)[0]

# %%
net7_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_7)
net17_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_17)
#net17_corr_connectomes.head()

# %%
corr_connectomes = get_connectomes(con_model='correlation_random')

# %%
lasso_connectomes, lasso_tp = get_connectomes(con_model='lassoBIC_blocks')

# %%
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)

# %%
#mask = [id in sublist for id in phenotype['participant_id']]
#rand100_phenotype = phenotype[phenotype[mask]]
#pprint.pprint(rand100_phenotype.loc)
#phenotype.set_index('participant_id', inplace=True)
rand100_phenotype = pd.DataFrame(columns=phenotype.columns)
for sub in sublist:
    rand100_phenotype = pd.merge(rand100_phenotype, phenotype.loc[[int(sub)]], how='outer')
rand100_phenotype.rename(columns={'id': 'participant_id'}, inplace=True)
rand100_phenotype.set_index('participant_id', inplace=True)
#pprint.pprint(rand100_phenotype)

# %%
#scikit score function? Yes each has a seperate one, BUT might need later
'''
def eval_metrics(X_train, y_train, X_test, y_test, model):
    """Calculates R2 scores for FC models."""

    test_rsq = r2_score(y_test, model.predict(X_test))
    train_rsq = r2_score(y_train, model.predict(X_train))

    return (test_rsq, train_rsq)
'''

# %%
#now I need to 1: make sure they are ordered correctly 
# (ideally just merge the dataframes on index... BUT with 2 missing them, need to
# exlude them ig, merge right?
lasso_data = pd.concat([lasso_connectomes, rand100_phenotype], join='inner', axis=1)
corr_data = pd.concat([corr_connectomes, rand100_phenotype], join='inner', axis=1)
net7_lasso_data = pd.concat([net7_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net17_lasso_data = pd.concat([net17_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net7_corr_data = pd.concat([net7_corr_connectomes, rand100_phenotype], join='inner', axis=1)
net17_corr_data = pd.concat([net17_corr_connectomes, rand100_phenotype], join='inner', axis=1)
net17_corr_data.head()

# %%
#TO DO: add something for a 0 fold thing for linear regression, along with just getting baseline tests

# %%
'''
#okay so want to plot these, BUT the thing is, the edges are different per fold... should do other stuff first
    if plot:
        f, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 5))
        ax1.scatter(sub_sums[], net7_lasso_results['predictions'], alpha=0.5)
        n7_b, n7_a = np.polyfit(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], 1)
        ax1.plot(net7_lasso_results['k_fold_age'], n7_a + n7_b * net7_lasso_data['age'], color='red')

        ax2.scatter(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], alpha=0.5)
        n17_b, n17_a = np.polyfit(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], 1)
        ax2.plot(net17_lasso_results['k_fold_age'], n17_a + n17_b * net17_lasso_data['age'], color='red')

        ax1.set_title('Positive Network')
        ax2.set_title('Negative Network')
        ax3.set_title('Total')
'''

# %%
'''
#OKAY SO need to reqork the cpm to not use sums, but just use the selected edges??? 
# So first get those labels, then just try ridge on them alone? seems good
# though i do have a big question... IF this worked so well for them, why is my cpm so bad??? why use sum scores?
# so this isn't working... idk if its worth making this work
X = lasso_data[corr_data.columns[:10000]]
y = lasso_data['age']
folds = 10
alpha = .01
model_obj = LinearRegression()
correlate = 'corr'
selection = 'negative'

results = {}
predictions = []
edge_labels = X.columns
y = np.array(y)
kf = KFold(folds)
fold = 0
k_fold_y = []
for train_index, test_index in kf.split(X):
    r_vals = []
    p_vals = []
    pos_net = set()
    neg_net = set()

    train_X = X.iloc[train_index]
    train_y = y[train_index]
    test_X = X.iloc[test_index]
    test_y = y[test_index]

    train_networks = {}
    test_networks = {}

    if correlate == 'corr':
        for edge in edge_labels:
            r, p = pearsonr(train_X[edge], train_y) #could make this skip the diagonal edges to get rid of the warning, but I dont think it 
            r_vals.append(r)
            p_vals.append(p)

    for r, p, edge in zip(r_vals, p_vals, edge_labels):
        if r > 0 and p < alpha:
            pos_net.add(edge)
        elif r < 0 and p < alpha:
            neg_net.add(edge)

    pos_cols = [edge for edge in edge_labels if edge in pos_net]
    neg_cols = [edge for edge in edge_labels if edge in neg_net]
    total_cols = pos_cols + neg_cols

    train_networks['positive'] = train_X[pos_cols]
    train_networks['negative'] = train_X[neg_cols]
    train_networks['total'] = train_X[total_cols]

    test_networks['positive'] = test_X[neg_cols]
    test_networks['negative'] = test_X[neg_cols]
    test_networks['total'] = test_X[total_cols]

    pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
    pl.fit(train_networks[selection], train_y)
    results[f'fold {fold} stats'] = {'test': pl.score(test_networks[selection], test_y), 'train': pl.score(train_networks[selection], train_y)}
    results[f'fold {fold} coef'] = model_obj.coef_
    k_fold_y.extend(test_y)
    predictions.extend(pl.predict(test_networks[selection]))
    fold += 1

results['k_fold_age'] = k_fold_y
results['full_r2'] = r2_score(k_fold_y, predictions)
results['predictions'] = predictions
    
    #could theoretically call my model fitting function here, if I make it work with 0 folds (could also additionally crossvalidate this step?)

    fit.model
    pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
    pl.fit(train_n_edges_X, train_y)
    results[f'fold {fold} stats'] = {'test': pl.score(test_n_edges_X, test_y), 'train': pl.score(train_n_edges_X, train_y)}
    predictions.extend(pl.predict(test_n_edges_X))
    #can refit and repredict for pos and neg networks, if either is better I would be very confused.
    k_fold_y.extend(test_y)
    print(f'fold {fold} complete')
    fold += 1

results['k_fold_age'] = k_fold_y
results['full_r2'] = r2_score(k_fold_y, predictions)
results['predictions'] = predictions
'''

# %%
#print(results['full_r2'])

# %%
cpm_corr_results = cpm(corr_data[corr_data.columns[:10000]], corr_data['age'], 5, 'corr', .01, LinearRegression(), True)
cpm_lasso_results = cpm(lasso_data[corr_data.columns[:10000]], lasso_data['age'], 5, 'corr', .01, LinearRegression(), False)

# %%
print(cpm_corr_results['full_r2'], 'r:', cpm_corr_results['r'][0, 1], 'mae', cpm_corr_results['MAE'])
print(cpm_lasso_results['full_r2'], 'r:', cpm_lasso_results['r'][0, 1], 'mae', cpm_lasso_results['MAE'])

# %%
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(cpm_corr_results['k_fold_age'], cpm_corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(cpm_corr_results['k_fold_age'], cpm_corr_results['predictions'], 1)
ax1.plot(cpm_corr_results['k_fold_age'], corr_a + corr_b * corr_data['age'], color='red')

ax2.scatter(cpm_lasso_results['k_fold_age'], cpm_lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(cpm_lasso_results['k_fold_age'], cpm_lasso_results['predictions'], 1)
ax2.plot(cpm_lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')

ax1.set_title('CPM age correlation')
ax2.set_title('CPM age lasso')
plt.show()

# %%
#I don't know if doing CPM on the network connectomes makes sense, usually people do the grouping afterwards from the positive and negative networks.
#how different are the sum values of people?
#would just negative work better? there are more of them, is that typical generally?

# %%
data = net7_lasso_data
#Recreating the RBC figure 7 stuff
vis_edge = 'SalVentAttn_to_SalVentAttn'
vis_dfn_edge = 'SalVentAttn_to_Default'
net7_labels = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Control', 'Default Mode']
net7_labels.reverse()
locations = np.arange(len(net7_labels))
#first just the matrix, want to see
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
age_matrix = []
for edge in net7_corr_data.columns[:49]:
    age_matrix.append(scipy.stats.pearsonr(data[edge], data['age'])[0])
age_matrix.reverse()
age_matrix = np.reshape(age_matrix, (7, 7))
#age_matrix = np.tril(age_matrix)

sns.heatmap(age_matrix, annot=True, cmap='bwr', fmt='.2f',ax=axs[0], linewidths=0.5, vmin=-0.25, vmax=0.25) 
axs[0].set_xticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_yticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_title('Network Connectivity Correlations to Age')


axs[1].scatter(data['age'], data[vis_edge], alpha=0.5)
n7_b, n7_a = np.polyfit(data['age'], data[vis_edge], 1)
axs[1].plot(data['age'], n7_a + n7_b * data['age'], color='darkorange')
axs[1].set_xlabel('age (years)')
axs[1].set_ylabel('LASSO Functional Connectivity (r)', rotation=90)
axs[1].set_title('Ventral Attention - Ventral Attention')
fig.tight_layout()
plt.show()
#ax.set_yticks(locations)
#ax.set_yticks(locations)
#ax.set_xticklabels(net7_labels)
#ax.set_yticklabels(net7_labels)

# %%
lasso_data.head()

# %%
# adding in bootstrapping
# NO KONAY SO SWITCH MY THING TO groupKFold, make the groups the ids. that should be fine.
# just running it 
# getting an SE, formula in the stat book, not in the skelarn module. or just save them to an array and get se?

# %%
resampled = lasso_data.sample(frac=1, replace=True)

# %%
#for data merging, move things out, be clearer, erorrs can happen there
folds = 5
#model = 
#model = 
#fracs = np.linspace(0, 1, n_alphas)
#model = FracRidgeRegressor(fracs=fracs, fit_intercept=True) #fits 20 predictions, one per alpha, sooooo would need to manually select the best alpha?
samples = 3
model_names = ['lassoCV', 'ridgeCV, LinearRegression', 'PCLasso']
results = {model:{} for model in model_names}
models = [LassoCV(random_state=random_state), RidgeCV(), LinearRegression(), LassoCV(random_state=random_state)]
datas = [lasso_data, corr_data, net7_lasso_data, net7_corr_data, net17_lasso_data, net17_corr_data]
length_of_connectomes = [10000, 10000, 49, 49, 289, 289]
for model_name, model in zip(model_names, models):
    for data, length in zip(datas, length_of_connectomes):
        r2s = []
        rs = []
        MAEs = []
        if model_name == 'PCLasso':
            component = 'PCA'
        else:
            component = None
        for i in range(samples):
            resample = lasso_data.sample(frac=1, replace=True)
            results[model_name][f'sample {i}'] = fit_model(data[[f'edge_{num}' for num in range(length)]], data['age'], model, folds=folds, component=component, verbose=True)

            r2s.append(results[model_name][f'sample {i}']['full_r2'])
            rs.append(results[model_name][f'sample {i}']['r'])
            MAEs.append(results[model_name][f'sample {i}']['MAE'])
        
        results[model_name]['full_r2'] = statistics.mean(r2s)
        results[model_name]['r2_CI'] = np.percentile(r2s, [2.5, 97.5])
        results[model_name]['r2_SD'] = statistics.stdev(r2s)
        
        results[model_name]['r'] = statistics.mean(rs)
        results[model_name]['r_CI'] = np.percentile(rs, [2.5, 97.5])
        results[model_name]['r_SD'] = statistics.stdev(rs)
        
        results[model_name]['MAE'] = statistics.mean(MAEs)
        results[model_name]['MAE_CI'] = np.percentile(MAEs, [2.5, 97.5])
        results[model_name]['MAE_SD'] = statistics.stdev(MAEs)

# %%
quarts = np.percentile(lasso_r2s, [2.5, 97.5])
print(quarts[1] - quarts[0])
print(lasso_results['r2_SD'] * 2)

# %%
print('lasso: r2:', lasso_results['full_r2'], 'r:', lasso_results['r'], 'MAE:', lasso_results['MAE']) #looks just like a lot of overfitting
print('correlation:', corr_results['full_r2'], 'r:', corr_results['r'], 'MAE:', corr_results['MAE']) #um huh, maybe just underfitting from not enough subjects??? weird b/c still lots of features
print('lasso 7 networks:', net7_lasso_results['full_r2'], 'r:', net7_lasso_results['r'], 'MAE:', net7_lasso_results['MAE'])
print('net17 lasso, r2:', net17_lasso_results['full_r2'], 'r:', net17_lasso_results['r'], 'MAE:', net17_lasso_results['MAE'])
print('correlation 7 networks:', net7_corr_results['full_r2'], 'r:', net7_corr_results['r'], 'MAE:', net7_corr_results['MAE'])
print('correlation 17 networks:', net17_corr_results['full_r2'], 'r:', net17_corr_results['r'], 'MAE:', net17_corr_results['MAE'])

# %%
pca_lasso_results = lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], model, folds=folds, component='PCA')
plt.scatter(pca_lasso_results['k_fold_age'], pca_lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(pca_lasso_results['k_fold_age'], pca_lasso_results['predictions'], 1)
plt.plot(pca_lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')

plt.title('PCA lasso age prediction')
plt.show()
print('lasso:', pca_lasso_results['fold 1 stats'], 'full r2:', pca_lasso_results['full_r2'])

# %%
#plotting residuals
plt.scatter(pca_lasso_results['k_fold_age'], pca_lasso_results['residuals'])

# %%
#for data merging, move things out, be clearer, erorrs can happen there
folds = 10
model = LassoCV(random_state=random_state)
#model = LinearRegression()
#fracs = np.linspace(0, 1, n_alphas)
#model = FracRidgeRegressor(fracs=fracs, fit_intercept=True) #fits 20 predictions, one per alpha, sooooo would need to manually select the best alpha?
#model = RidgeCV()
pca_lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], model, folds=folds, component='PCA')
#pca_corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], model, folds=folds, component='PCA')

# %%
print('lasso:', pca_lasso_results['fold 1 stats'], 'full r2:', pca_lasso_results['full_r2']) #looks just like a lot of overfitting
#print('correlation:', pca_corr_results['fold 1 stats'], 'full r2:', pca_corr_results['full_r2'])

# %%
#TO DO: correlate residuals with p values??? at least plot them... if I see outliers like their 10% i could remove those

# %%
#TO DO: get the nilearn correlaiton data visualizations working??? (the cai figure 3) (this is definetly low on priorities rn though)

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
