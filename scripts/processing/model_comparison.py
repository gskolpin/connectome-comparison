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
con_model = 'lassoBIC_blocks'
#filepath for the predictor or just all the phenotype data (need to make a space for those in scrubbed)
p_filepath = '/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv'
scaler = StandardScaler() #making it normal so things work with it
sublist_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/rand709_sub_list.txt'
motion_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/pnc_motion.tsv'

# %%
p = 'p_factor_mcelroy_harmonized_all_samples'

# %%
sub_list = []
with open(sublist_filepath, 'r') as file:
    sub_list = file.read().splitlines()
sub_list = [int(s) for s in sub_list]
print(len(sub_list))

# %%
atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_filename = atlas.maps
#plot_roi(atlas_filename, title="Schaefer_2018 atlas", view_type="contours")
    
atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas17_filename = atlas.maps

# %%
full_labels = atlas.labels[1:]
labels_7 = [label.split("_")[2] for label in full_labels]

full_labels_17 = atlas_17.labels[1:]
labels_17 = [label.split("_")[2] for label in full_labels_17]

# %%
lasso_connectomes = get_networks(con_model=con_model, network_labels=labels_7)
net17_lasso_connectomes = get_networks(con_model=con_model, network_labels=labels_17)

# %%
corr_connectomes = get_networks(con_model='correlation_random', network_labels=labels_7)
net17_corr_connectomes = get_networks(con_model='correlation_random', network_labels=labels_17)

# %%
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)

motion = pd.read_csv(motion_filepath, delimiter = '\t', header=0)
motion['sub'] = motion['sub'].str[4:]
motion['sub'] = pd.to_numeric(motion['sub'], errors='coerce').fillna(0).astype(int)
motion.set_index('sub', inplace=True, drop=False)
motion.rename(columns={'sub': 'id'}, inplace=True)
motion = motion[~motion.index.duplicated(keep='first')]

# %%
sample_phenotype = phenotype.loc[sub_list]
sample_motion = motion.loc[sub_list]

sample_phenotype.rename(columns={'id': 'participant_id'}, inplace=True)
sample_phenotype.set_index('participant_id', inplace=True)
sample_motion.rename(columns={'id': 'participant_id'}, inplace=True)
sample_motion.set_index('participant_id', inplace=True)
sample_motion = sample_motion[~sample_motion.index.duplicated(keep='first')]

# %%
lasso_data = pd.concat([lasso_connectomes, sample_phenotype, sample_motion['meanFD']], join='inner', axis=1)
corr_data = pd.concat([corr_connectomes, sample_phenotype, sample_motion['meanFD']], join='inner', axis=1)

net17_lasso_data = pd.concat([net17_lasso_connectomes, sample_phenotype], join='inner', axis=1)
net17_corr_data = pd.concat([net17_corr_connectomes, sample_phenotype], join='inner', axis=1)

# %%
#drop values from corr data that are dropped from lasso data for not having enough data
subs_to_drop = list(set(corr_data.index) - set(lasso_data.index))
corr_data.drop(subs_to_drop, inplace=True, axis=0)
net17_corr_data.drop(subs_to_drop, inplace=True)

datas = [lasso_data, corr_data, net17_lasso_data, net17_corr_data]

#drop nan values in the 
for data in datas:
    data.dropna(axis=0, inplace=True)

# %%
print(lasso_data.columns)

# %%
#renameing the long name columns
stats = ['attention',
         'externalizing',
         'internalizing',
         'p_factor']
mapper = {f'{stat}_mcelroy_harmonized_all_samples': stat for stat in stats}
for data in datas:
    data.rename(mapper, axis=1, inplace=True)

# %%
lasso_data.head()

# %%
#first want to just visualize the correlations with both p and attention again just to look

# %%
#making font in them bigger
#these can be smaller
# MCKENZIES CODE for centering the label markers
    # and colors 
predictors = ['age', 'p_factor', 'attention']
datas = [lasso_data, corr_data]
data_names = ['Lasso networks', 'corr networks']

net7_labels = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Control', 'Default Mode']
net7_labels = list(reversed(net7_labels))
locations = np.arange(len(net7_labels))

fig, axs = plt.subplots(3, 2, figsize=(12, 15))

for i, predictor in zip(range(len(predictors)), predictors):
    for j, data, name in zip(range(len(datas)), datas, data_names):

        corr_vals = []
        for edge in data.columns[:49]:
            r = scipy.stats.pearsonr(data[edge], data[predictor])[0]
            corr_vals.append(r)
        corr_matrix = np.reshape(corr_vals, (7, 7))
        if 'corr' in name:
            corr_matrix = np.tril(corr_matrix)
        ax = axs[i, j]
        sns.heatmap(corr_matrix,
            ax=ax,
            cmap='bwr',
            vmin=-0.25, vmax=0.25,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            xticklabels=net7_labels,
            yticklabels=net7_labels)

        ax.set_title(f"{predictor} × {data_names[j]}")
        ax.set_xticks(locations)
        ax.set_yticks(locations)
        ax.set_xticklabels(net7_labels, rotation=45, ha='right')
        ax.set_yticklabels(net7_labels, rotation=0)

plt.tight_layout()
plt.show()

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
covariates = ['age', 'meanFD']
dmn_edges = [edge for edge in lasso_data.columns if 'Default' in edge]
control_edges = [edge for edge in lasso_data.columns if 'Cont' in edge]
ven_attention_edges = [edge for edge in lasso_data.columns if 'SalVentAttn' in edge]
dors_attention_edges = [edge for edge in lasso_data.columns if 'DorsAttn' in edge]

# %%
nets_of_int = {"Default", "SalVentAttn", "DorsAttn", "Cont"}
int_net_edges = [name for name in lasso_data.columns
    if (parts := name.split("_to_")) and parts[0] in nets_of_int and parts[1] in nets_of_int]

# %%
dmn_edges.extend(covariates)
control_edges.extend(covariates)
ven_attention_edges.extend(covariates)
dors_attention_edges.extend(covariates)
int_net_edges.extend(covariates)

# %%
#checking that the network masking works
corr_data[int_net_edges].head()

# %%
#Need to take notes rn
'''
- don't need to use the age covariate regressed data, simpler to just add it to linear models
- the 4 picks for networks are mainly dmn, dorsal attention, ventral attention, and control (maybe somatomotor??? but this is more attention and not physical impulsivity... eh... idk)
- I think theres an interesting comparison to be done with all of the other networks as well right? like if the differences in those networks to others is lower in corr than in lasso, 
  - because corr is more correlated generally... does that make sense?
- on the graphs, limbic also stands out just in lasso though
'''

# %%
#will also want to run it with p, would be interesting to compare
predictor = 'p_factor'

# %%
cov_formula = f"{predictor} ~ " + " + ".join(covariates)
dmn_formula = f"{predictor} ~ " + " + ".join(dmn_edges)
control_formula = f"{predictor} ~ " + " + ".join(control_edges)
ven_attention_formula = f"{predictor} ~ " + " + ".join(ven_attention_edges)
dors_attention_formula = f"{predictor} ~ " + " + ".join(dors_attention_edges)
int_formula = f"{predictor} ~ " + " + ".join(int_net_edges)
stripped_lasso_model= smf.ols(cov_formula, lasso_data).fit()
#stripped_corr_model
datas = [lasso_data, corr_data]
data_names = ['lasso', 'corr']
formulas = [dmn_formula, control_formula, ven_attention_formula, dors_attention_formula, int_formula]
network_names = ['dmn', 'control', 'ven_attention', 'dors_attention', 'networks_of_note']

models = {name: {} for name in data_names}
for data, name in zip(datas, data_names):
    for formula, network in zip(formulas, network_names):
        models[name][network] = smf.ols(formula, data).fit()

# %%
#just set the network here
net = 'networks_of_note'

# %%
sm.stats.anova_lm(models['lasso'][net], typ=2)

# %%
sm.stats.anova_lm(models['corr'][net], typ=2)

# %%
print(int_net_edges)

# %%
#also want to try reversing the direction.
#set edges of interest very first
int_edges = lasso_data.columns[:49]
#int_edges = int_net_edges

model_params = ['age', 'meanFD', predictor]
formulas = {}
for edge in int_edges:
    formulas[edge] = dmn_formula = f"{edge} ~ " + " + ".join(model_params)

models = {name: {} for name in data_names}
for data, name in zip(datas, data_names):
    for network, formula in formulas.items():
        models[name][network] = smf.ols(formula, data).fit()

# %%
#asign data used, and alpha cuttoff:
#VERY interesting here to see differences in lasso and corr when looking at the all networks, a result I do expect there (ish, kinda, sorta)!

#here could theoretically use scipy.stats.false_discovery_control() on the list of p factor things, and that might just give the adjusted p values?
#I thought it gave different alphas per p, but I guess the adjustment can be made either side.
data = 'lasso'
alpha = .05

for edge in int_edges:
    anova = sm.stats.anova_lm(models[data][edge])
    if anova.loc[predictor, 'PR(>F)'] <= alpha:
        print(edge, anova.loc[predictor, 'PR(>F)'])

# %%
#asign data used, and alpha cuttoff:
#VERY interesting here to see differences in lasso and corr when looking at the all networks, a result I do expect there (ish, kinda, sorta)!
data = 'corr'
alpha = .05

for edge in int_edges:
    anova = sm.stats.anova_lm(models[data][edge])
    if anova.loc[predictor, 'PR(>F)'] <= alpha:
        print(edge, anova.loc[predictor, 'PR(>F)'])

# %%
