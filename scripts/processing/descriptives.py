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

from fracridge import fracridge, FracRidgeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from functions import get_connectomes
from functions import cpm
from functions import fit_model
from functions import get_networks
from functions import get_timeseries_pred

# %load_ext autoreload
# %autoreload 2

# %%
# Ideally everything hardcoded here, so can just switch these
#filepath to the csv connectivity files themselves
con_filepath = '/gscratch/scrubbed/gkolpin/xcpd_output/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd'
#and model
con_model = 'lassoBIC_task'
#filepath for the predictor or just all the phenotype data (need to make a space for those in scrubbed)
p_filepath = '/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv'
scaler = StandardScaler() #making it normal so things work with it
sublist_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/rand709_sub_list.txt'
random_state = 10
networks = True

# %%
sublist = []
with open(sublist_filepath, 'r') as file:
    sublist = file.read().splitlines()
print(len(sublist))

# %%
if networks:
    atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
    atlas_filename = atlas.maps
    plot_roi(atlas_filename, title="Schaefer_2018 atlas", view_type="contours")
    
    atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
    atlas17_filename = atlas.maps

# %%
if networks:
    full_labels = atlas.labels[1:]
    labels_7 = [label.split("_")[2] for label in full_labels]
    
    full_labels_17 = atlas_17.labels[1:]
    labels_17 = [label.split("_")[2] for label in full_labels_17]

# %%
if networks:
    net7_lasso_connectomes = get_networks(con_model=con_model, network_labels=labels_7)
    net17_lasso_connectomes = get_networks(con_model=con_model, network_labels=labels_17)
lasso_connectomes = get_connectomes(con_model=con_model)

# %%
if networks:
    net7_corr_connectomes = get_networks(con_model='correlation_random', network_labels=labels_7)
    net17_corr_connectomes = get_networks(con_model='correlation_random', network_labels=labels_17)
corr_connectomes = get_connectomes(con_model='correlation_random')

# %%
#wanting to visualize the connectomes, just get a view, and ideally view the network grouped ones too
  #should do this through nilearn, Also should combine all of them with labels into just one figure
f, axs = plt.subplots(1, 2)
ids = lasso_connectomes.index
c1 = axs[0].imshow(np.reshape(lasso_connectomes.loc[ids[0]], (100, 100)), cmap='bwr')
c2 = axs[1].imshow(np.reshape(corr_connectomes.loc[ids[0]], (100, 100)), cmap='bwr')
cbar = f.colorbar(c1, ax=axs, shrink=0.5)

# %%
#okay so the correlation one does not look too off, actually kinda makes sense
#need to add axis names for what network is where
f, axs = plt.subplots(1, 4)
ids = lasso_connectomes.index
c1 = axs[0].imshow(np.reshape(net7_lasso_connectomes.loc[ids[0]], (7, 7)), vmin=-1, vmax=1, cmap='bwr')
c2 = axs[1].imshow(np.reshape(net17_lasso_connectomes.loc[ids[0]], (17, 17)), vmin=-1, vmax=1, cmap='bwr')
c3 = axs[2].imshow(np.reshape(net7_corr_connectomes.loc[ids[0]], (7, 7)), vmin=-1, vmax=1, cmap='bwr')
c3 = axs[3].imshow(np.reshape(net17_corr_connectomes.loc[ids[0]], (17, 17)), vmin=-1, vmax=1, cmap='bwr')
f.tight_layout()
cbar = f.colorbar(c1, ax=axs, shrink=0.5)

# %%
#so connectomes look good but it seems like im only getting 98 participants?
#here ill figure out who im missing
for sub in sublist:
    if np.int64(sub) not in lasso_connectomes.index:
        print(sub)
#ya so these I cannot run, getting this error message:
'''
ValueError: You are using LassoLarsIC in the case where the number of samples is
smaller than the number of features. In this setting, getting a good estimate for
the variance of the noise is not possible. Provide an estimate of the noise
variance in the constructor.
'''
#I will just exclude them for now, but this might be solved once we get the task scans going

# %%
#getting the average connectome and SD
avg_lasso_connectome = lasso_connectomes.mean()
avg_corr_connectome = corr_connectomes.mean()
avg_net7_lasso_connectome = net7_lasso_connectomes.mean()
avg_net7_corr_connectome = net7_corr_connectomes.mean()
avg_net17_lasso_connectome = net17_lasso_connectomes.mean()
avg_net17_corr_connectome = net17_corr_connectomes.mean()
f, axs = plt.subplots(3, 2)
ids = lasso_connectomes.index
c1 = axs[0, 0].imshow(np.reshape(avg_lasso_connectome, (100, 100)), vmin=-1, vmax=1, cmap='bwr')
c2 = axs[0, 1].imshow(np.reshape(avg_corr_connectome, (100, 100)), vmin=-1, vmax=1, cmap='bwr')
c3 = axs[1, 0].imshow(np.reshape(avg_net7_lasso_connectome, (7, 7)), vmin=-1, vmax=1, cmap='bwr')
c4 = axs[1, 1].imshow(np.reshape(avg_net7_corr_connectome, (7, 7)), vmin=-1, vmax=1, cmap='bwr')
c5 = axs[2, 0].imshow(np.reshape(avg_net17_lasso_connectome, (17, 17)), vmin=-1, vmax=1, cmap='bwr')
c6 = axs[2, 1].imshow(np.reshape(avg_net17_corr_connectome, (17, 17)), vmin=-1, vmax=1, cmap='bwr')

axs[0, 0].set_title('Lasso Average Connectome', fontsize=10)
axs[0, 1].set_title('Pearson Average Connectome', fontsize=10)
for ax_col in axs:
    for ax in ax_col:
        ax.set_xticks([])
        ax.set_yticks([])
cbar = f.colorbar(c1, ax=axs, shrink=0.5)
plt.savefig('avg_connectomes')

# %%
# sd ones are on a different scale so I think need to be plotted differently
sd_lasso_connectome = lasso_connectomes.std()
sd_corr_connectome = corr_connectomes.std()
sd_net7_lasso_connectome = net7_lasso_connectomes.std()
sd_net7_corr_connectome = net7_corr_connectomes.std()
sd_net17_lasso_connectome = net17_lasso_connectomes.std()
sd_net17_corr_connectome = net17_corr_connectomes.std()
ids = lasso_connectomes.index
f, axs = plt.subplots(3, 2)
c1 = axs[0, 0].imshow(np.reshape(sd_lasso_connectome, (100, 100)), vmin=0, vmax=.35, cmap='binary')
c2 = axs[0, 1].imshow(np.reshape(sd_corr_connectome, (100, 100)), vmin=0, vmax=.35, cmap='binary')
c3 = axs[1, 0].imshow(np.reshape(sd_net7_lasso_connectome, (7, 7)), vmin=0, vmax=.35, cmap='binary')
c4 = axs[1, 1].imshow(np.reshape(sd_net7_corr_connectome, (7, 7)), vmin=0, vmax=.35, cmap='binary')
c5 = axs[2, 0].imshow(np.reshape(sd_net17_lasso_connectome, (17, 17)), vmin=0, vmax=.35, cmap='binary')
c6 = axs[2, 1].imshow(np.reshape(sd_net17_corr_connectome, (17, 17)), vmin=0, vmax=.35, cmap='binary')

axs[0, 0].set_title('Lasso Connectomes Standard Deviation', fontsize=10, rotation=10)
axs[0, 1].set_title('Pearson Connectomes Standard Deviation', fontsize=10, rotation=10)
for ax_col in axs:
    for ax in ax_col:
        ax.set_xticks([])
        ax.set_yticks([])
cbar = f.colorbar(c1, ax=axs, shrink=0.5)

# %%
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)

# %%
#looking at sparcity of connectomes Ive got
#TO DO: maybe something thats the opposite of sparsity, like looking at generally strong edges? 
cuttoff = 0.23 #max for task is around here
sparce_edges = {}
diagonal_edges = {}
for edge_name, edges in lasso_connectomes.items():
    sparcity = (len(edges) - np.count_nonzero(edges)) / len(edges)
    if sparcity >= 0.9:
        diagonal_edges[edge_name] = sparcity
    elif sparcity >= cuttoff:
        sparce_edges[edge_name] = sparcity

print(len(sparce_edges))
pprint.pprint(sparce_edges)

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
lasso_data = pd.concat([lasso_connectomes, rand100_phenotype], join='inner', axis=1)
corr_data = pd.concat([corr_connectomes, rand100_phenotype], join='inner', axis=1)
net7_lasso_data = pd.concat([net7_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net17_lasso_data = pd.concat([net17_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net7_corr_data = pd.concat([net7_corr_connectomes, rand100_phenotype], join='inner', axis=1)
net17_corr_data = pd.concat([net17_corr_connectomes, rand100_phenotype], join='inner', axis=1)

# %%
fullXs = [corr_data[corr_data.columns[:10000]], lasso_data[lasso_data.columns[:10000]]]
fullys = [corr_data['age'], lasso_data['age']]
X7s = [net7_corr_data[net7_corr_connectomes.columns], net7_lasso_data[net7_lasso_connectomes.columns]]
y7s = [net7_corr_data['age'], net7_lasso_data['age']]
X17s = [net17_corr_data[net17_corr_connectomes.columns], net17_lasso_data[net17_lasso_connectomes.columns]]
y17s = [net17_corr_data['age'], net17_lasso_data['age']]
allXs = [fullXs, X7s, X17s]
allys = [fullys, y7s, y17s]

n_alphas = 20
alphas = np.logspace(-10, 10, n_alphas)
rr_alphas = np.logspace(-10, 10, n_alphas)

fig, ax = plt.subplots(3, 2, figsize=(12, 16))
titles = ['Correlation Full', 'Lasso Full', 'Correlation 7 Network', 'Lasso 7 Network', 'Correlation 17 Network', 'Lasso 17 Network']
t = 0

for row, Xs, ys in zip(ax, allXs, allys):
    for col, X, y in zip(row, Xs, ys):
        rr_coefs = []
        rr_coefs = np.zeros((X.shape[-1], n_alphas))
        rr_pred = np.zeros((y.shape[-1], n_alphas))
        fracs = np.linspace(0, 1, n_alphas)

        FR = FracRidgeRegressor(fracs=fracs, fit_intercept=True)
        FR.fit(X, y)
        fr_pred = cross_val_predict(FR, X, y)
    
        col.plot(fracs, FR.coef_.T)
        ylims = col.get_ylim()
        col.vlines(fracs, ylims[0], ylims[1], linewidth=0.5, color='gray')
        col.set_ylim(*ylims)
        col.set_title(titles[t])
        t += 1
plt.tight_layout()
plt.show()

# %%
#if I want to do p things
p = 'p_factor_mcelroy_harmonized_all_samples'
datas = [lasso_data, corr_data, net7_lasso_data, net7_corr_data, net17_lasso_data, net17_corr_data]
for data in datas:
    data.dropna(axis=0, subset=[p], inplace=True)

# %%
#Recreating the RBC figure 7 stuff
data = net7_corr_data
predictor = 'age'
vis_edge = 'SalVentAttn_to_SalVentAttn'
vis_dfn_edge = 'SalVentAttn_to_Default'
net7_labels = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Control', 'Default Mode']
net7_labels.reverse()
locations = np.arange(len(net7_labels))
#first just the matrix, want to see
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
age_matrix = []
for edge in net7_corr_data.columns[:49]:
    age_matrix.append(scipy.stats.pearsonr(data[edge], data[predictor])[0])
age_matrix.reverse()
age_matrix = np.reshape(age_matrix, (7, 7))
#age_matrix = np.tril(age_matrix)

sns.heatmap(age_matrix, annot=True, cmap='bwr', fmt='.2f',ax=axs[0], linewidths=0.5, vmin=-0.25, vmax=0.25) 
axs[0].set_xticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_yticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_title('Network Connectivity Correlations to ' + predictor)


axs[1].scatter(data[predictor], data[vis_edge], alpha=0.5)
n7_b, n7_a = np.polyfit(data[predictor], data[vis_edge], 1)
axs[1].plot(data[predictor], n7_a + n7_b * data[predictor], color='darkorange')
axs[1].set_xlabel(predictor)
axs[1].set_ylabel('LASSO Functional Connectivity (r)', rotation=90)
axs[1].set_title('Ventral Attention - Ventral Attention')
fig.tight_layout()
plt.show()
#ax.set_yticks(locations)
#ax.set_yticks(locations)
#ax.set_xticklabels(net7_labels)
#ax.set_yticklabels(net7_labels)

# %%
histage = rand100_phenotype.hist(['age'], bins=np.arange(32) - 0.5)
plt.xticks(range(8, 24))
plt.xlim(7, 24)
plt.grid(False)
plt.xlabel('age')
plt.ylabel('count')
plt.title('PNC age distribution')
plt.show()
print('min:',  rand100_phenotype['age'].min(), 'max:', rand100_phenotype['age'].max())

# %%
histage = rand100_phenotype.hist(['p_factor_mcelroy_harmonized_all_samples'])
#plt.xticks(range(8, 24))
#plt.xlim(7, 24)
plt.grid(False)
plt.xlabel('p-factor')
plt.ylabel('count')
plt.title('PNC p-factor distribution')
plt.show()
print('min:',  rand100_phenotype[p].min(), 'max:', rand100_phenotype[p].max())

# %%
histage = rand100_phenotype.hist(['attention_mcelroy_harmonized_all_samples'])
#plt.xticks(range(8, 24))
#plt.xlim(7, 24)
plt.grid(False)
plt.xlabel('attention')
plt.ylabel('count')
plt.title('PNC attention distribution')
plt.show()
print('min:',  rand100_phenotype['attention_mcelroy_harmonized_all_samples'].min(), 'max:', rand100_phenotype['attention_mcelroy_harmonized_all_samples'].max())

# %%
i = 0
for val in rand100_phenotype['p_factor_mcelroy_harmonized_all_samples']:
    if val == rand100_phenotype[p].min():
        i += 1
i

# %%
"just age with p factor plotted"
plt.scatter(lasso_data['age'], lasso_data[p])
lasso_b, lasso_a = np.polyfit(lasso_data['age'], lasso_data[p], 1)
plt.plot(lasso_data['age'], lasso_a + lasso_b * lasso_data['age'], color='red')

# %%
#looking at the prediciton accuracy but for regular not task
block_timeseries_pred = get_timeseries_pred(con_model='lassoBIC_blocks')
print(len(block_timeseries_pred))

# %%
#looking at the prediciton accuracy
task_timeseries_pred = get_timeseries_pred(con_model='lassoBIC_task')
print(len(task_timeseries_pred))

# %%
#This is averaged across node, so looking at the same people in each y. Want to do the inverse, averaged across people looking at node per y.
plt.scatter(task_timeseries_pred.keys(), task_timeseries_pred.values(), color='red', alpha=.5, label='task')
plt.scatter(block_timeseries_pred.keys(), block_timeseries_pred.values(), color='blue', alpha=.5, label='blocks')
plt.legend()
plt.title('timeseries prediction accuracy averaged across node')
#plt.tick_params(labelbottom=True)
plt.tight_layout()

# %%
