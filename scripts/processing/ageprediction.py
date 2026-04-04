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
import argparse
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--seed',default='1')
parser.add_argument('--con_filepath', default='/gscratch/scrubbed/gkolpin/xcpd_output/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd')
parser.add_argument('--sublist_filepath', default='/gscratch/escience/gkolpin/connectome-comparison/data/rand709_sub_list.txt')
parser.add_argument('--phenotype_filepath', default='/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv')
parser.add_argument('--label', default='test')

try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
  
seed = int(args.seed)
con_filepath = args.con_filepath
sublist_filepath = args.sublist_filepath
p_filepath = args.phenotype_filepath
label = args.label
print(args)

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
net7_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_7)[0]
net17_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_17)[0]
lasso_connectomes, lasso_tp = get_connectomes(con_model='lassoBIC_blocks')

# %%
net7_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_7)
net17_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_17)
corr_connectomes = get_connectomes(con_model='correlation_random')

# %%
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)

# %%
rand100_phenotype = pd.DataFrame(columns=phenotype.columns)
for sub in sublist:
    rand100_phenotype = pd.merge(rand100_phenotype, phenotype.loc[[int(sub)]], how='outer')
rand100_phenotype.rename(columns={'id': 'participant_id'}, inplace=True)
rand100_phenotype.set_index('participant_id', inplace=True)
#pprint.pprint(rand100_phenotype)

# %%
#scikit score function? Yes each has a seperate one, BUT might need later for UoI
'''
def eval_metrics(X_train, y_train, X_test, y_test, model):
    """Calculates R2 scores for FC models."""

    test_rsq = r2_score(y_test, model.predict(X_test))
    train_rsq = r2_score(y_train, model.predict(X_train))

    return (test_rsq, train_rsq)
'''

# %%
lasso_data = pd.concat([lasso_connectomes, rand100_phenotype], join='inner', axis=1)
corr_data = pd.concat([corr_connectomes, rand100_phenotype], join='inner', axis=1)
net7_lasso_data = pd.concat([net7_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net17_lasso_data = pd.concat([net17_lasso_connectomes, rand100_phenotype], join='inner', axis=1)
net7_corr_data = pd.concat([net7_corr_connectomes, rand100_phenotype], join='inner', axis=1)
net17_corr_data = pd.concat([net17_corr_connectomes, rand100_phenotype], join='inner', axis=1)
#net17_corr_data.head()

# %%
'''
#This section basically does CPM but, rather than using sum scores, keeps selected edges seperate. Does worse.
#Also, theoretically Lasso should be doing something very similar, so no real reason to tamper with this to get it to work.
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
all_model_names = ['ridgeCV', 'lassoCV', 'LinearRegression', 'PCLasso', 'cpm']
data_names = ['lasso', 'corr', 'net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr']
results = {model:{data:{} for data in data_names} for model in all_model_names}

datas = [lasso_data, corr_data, net7_lasso_data, net7_corr_data, net17_lasso_data, net17_corr_data]
length_of_connectomes = [[f'edge_{num}' for num in range(10000)], 
                         [f'edge_{num}' for num in range(10000)], 
                         net7_lasso_connectomes.columns[:49], 
                         net7_corr_data.columns[:49], 
                         net17_lasso_connectomes.columns[:289], 
                         net17_corr_connectomes.columns[:289]]

for data, data_name, length in zip(datas, data_names, length_of_connectomes):
    resample = data.sample(frac=1, replace=True)
    results['cpm'][data_name] = cpm(resample[length], resample['age'], 5, 'corr', .05, LinearRegression(), ('corr' in data_name))

# %%
folds = 5
model_names = ['ridgeCV', 'lassoCV', 'LinearRegression', 'PCLasso']
data_names = ['lasso', 'corr', 'net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr']
models = [RidgeCV(), LassoCV(random_state=seed), LinearRegression(), LassoCV(random_state=seed)]
datas = [lasso_data, corr_data, net7_lasso_data, net7_corr_data, net17_lasso_data, net17_corr_data]
length_of_connectomes = [[f'edge_{num}' for num in range(10000)], 
                         [f'edge_{num}' for num in range(10000)], 
                         net7_lasso_connectomes.columns[:49], 
                         net7_corr_data.columns[:49], 
                         net17_lasso_connectomes.columns[:289], 
                         net17_corr_connectomes.columns[:289]]

for data, data_name, length in zip(datas, data_names, length_of_connectomes):
    resample = data.sample(frac=1, replace=True)
    for model_name, model in zip(model_names, models):
        if model_name == 'PCLasso':
            component = 'PCA'
        else:
            component = None
        results[model_name][data_name] = fit_model(resample[length], resample['age'], model, folds=folds, component=component, verbose=False)

# %%
results_path = '/gscratch/scrubbed/gkolpin/age_predictions'
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
result_file = f'{task_id}_results.pkl'
job_dir = f'batch_{label}'
os.makedirs(op.join(results_path, job_dir), exist_ok=True)

with open(op.join(results_path, job_dir, result_file), 'wb') as f:
    pkl.dump(results, f) 

args_dict = vars(args)
config_file = result_file.replace('_results.pkl', '_results.json')
with open(op.join(results_path, job_dir, config_file), "w") as f:
    json.dump(args_dict, f, indent=4)
