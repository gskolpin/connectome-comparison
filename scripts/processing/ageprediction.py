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
from sklearn.multioutput import MultiOutputRegressor

from fracridge import fracridge, FracRidgeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from functions import get_connectomes
from functions import get_networks
from functions import get_timeseries_pred
from functions import cpm
from functions import fit_model
from functions import CovariateRegressor

# %load_ext autoreload
# %autoreload 2

# %%
import argparse
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--seed',default='1')
parser.add_argument('--con_filepath', default='/gscratch/psych/gkolpin/data/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd')
parser.add_argument('--sub_list_filepath', default='/gscratch/escience/gkolpin/connectome-comparison/data/1600_sub_list.txt')
parser.add_argument('--phenotype_filepath', default='/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv')
parser.add_argument('--motion_filepath', default='/gscratch/escience/gkolpin/connectome-comparison/data/pnc_motion.tsv')
parser.add_argument('--con_model', default='lassoBIC_task')
parser.add_argument('--networks', default=False)
parser.add_argument('--predictor', default='age')
parser.add_argument('--covariates', default=None)
parser.add_argument('--covariate_regressor', default=None)
parser.add_argument('--cpm', default='total')
parser.add_argument('--label', default='test')

try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
  
seed = int(args.seed)
con_filepath = args.con_filepath
sub_list_filepath = args.sub_list_filepath
p_filepath = args.phenotype_filepath
m_filepath = args.motion_filepath
con_model = args.con_model
networks = args.networks
predictor = args.predictor
covariates = args.covariates
covariate_regressor = args.covariate_regressor
cpm_type = args.cpm
label = args.label
print(args)

# %%
#So getting weights for PCLasso seems really anoying so im just gonna do this instead, if false it will run it, if true not.
if cpm_type is None:
    weights = True
else:
    weights=False

# %%
sub_list = []
with open(sub_list_filepath, 'r') as file:
    sub_list = file.read().splitlines()
sub_list = [int(s) for s in sub_list]

# %%
if networks:
    atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
    atlas_filename = atlas.maps
    #plot_roi(atlas_filename, title="Schaefer_2018 atlas", view_type="contours")
    
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
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)

motion = pd.read_csv(m_filepath, delimiter = '\t', header=0)
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
lasso_data = pd.concat([lasso_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)
corr_data = pd.concat([corr_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)
if networks:
    net7_lasso_data = pd.concat([net7_lasso_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)
    net17_lasso_data = pd.concat([net17_lasso_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)
    net7_corr_data = pd.concat([net7_corr_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)
    net17_corr_data = pd.concat([net17_corr_connectomes, sample_phenotype, sample_motion[['meanFD']]], join='inner', axis=1)

# %%
data_names = ['lasso', 'corr']
datas = [lasso_data, corr_data]
connectome_column_labels = [[f'edge_{num}' for num in range(len(lasso_connectomes.columns))], 
                         [f'edge_{num}' for num in range(len(corr_connectomes.columns))]] 
if networks:
    net7_length = 7**2
    net17_length = 17**2
    data_names.extend(['net7_lasso', 'net7_corr', 'net17_lasso', 'net17_corr'])
    datas.extend([net7_lasso_data, net7_corr_data, net17_lasso_data, net17_corr_data])
    connectome_column_labels.extend([net7_lasso_connectomes.columns[:net7_length], 
                              net7_corr_data.columns[:net7_length], 
                              net17_lasso_connectomes.columns[:net17_length], 
                              net17_corr_connectomes.columns[:net17_length]])

# %%
#drop values from corr data that are dropped from lasso data for not having enough data
subs_to_drop = list(set(corr_data.index) - set(lasso_data.index))
corr_data.drop(subs_to_drop, inplace=True)
if networks:
    net7_corr_data.drop(subs_to_drop, inplace=True)
    net17_corr_data.drop(subs_to_drop, inplace=True)

#drop nan values in the 
for data, name in zip(datas, data_names):
    data.dropna(axis=0, subset=[predictor], inplace=True)

# %%
if covariates is not None:
    if isinstance(covariates, str):
        covariates = [covariates]
    for data, labels, name in zip(datas, connectome_column_labels, data_names):
        for covariate in covariates:
            data = data.dropna(axis=0, subset=[covariate])
            data_cov = data[covariate].to_numpy(dtype=float)
            data_X = data[labels].to_numpy(dtype=float)
            
        covreg = CovariateRegressor(
            covariate=data_cov,
            X_full=data_X,
            estimator=covariate_regressor,
            cross_validate=True
        )

        new_data = {}
        residual_data = covreg.fit_transform(data_X)
        for sub, new_connectome in zip(sub_list, residual_data):
            new_data[int(sub)] = new_connectome

        new_data = pd.DataFrame(new_data).transpose()
        edge_mapping = {i: data.columns[i] for i in range(len(data.columns))}
        new_data = new_data.rename(columns=edge_mapping)
        data[new_data.columns] = new_data

# %%
folds = 5
#Does not have cpm or PClasso because of weights... also those seemed like less impactful results so oh well...
model_names = ['ridgeCV', 'lassoCV', 'LinearRegression']
models = [RidgeCV(), LassoCV(random_state=seed), LinearRegression()]
if cpm_type is not None: 
    model_names.append('PCLasso')
    model_names.append('PCReg')
    models.append(LassoCV(random_state=seed))
    models.append(LinearRegression())
results = {model:{data:{} for data in data_names} for model in model_names}

# %%
if cpm_type is not None:
    results['cpm'] = {}
    for data, data_name, length in zip(datas, data_names, connectome_column_labels):
        resample = data.sample(frac=1, replace=True)
        results['cpm'][data_name] = cpm(resample[length], resample[predictor], 5, 'corr', .05, cpm_type, LinearRegression(), ('corr' in data_name))

# %%
for data, data_name, length in zip(datas, data_names, connectome_column_labels):
    resample = data.sample(frac=1, replace=True)
    for model_name, model in zip(model_names, models):
        if 'PC' in model_name:
            component = 'PCA'
        else:
            component = None
        results[model_name][data_name] = fit_model(resample[length], resample[predictor], model, folds=folds, component=component, verbose=False, weights=weights)

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

# %%
#do I keep the test here???
from sklearn.model_selection import train_test_split
"""
Test basic covariate regression with default parameters.

Scenario: Standard use case with multiple covariates and default settings
(cross_validate=True, stack_intercept=True).
"""
np.random.seed(42)
n_samples, n_features = 100, 20

# Create synthetic data
X = np.random.randn(n_samples, n_features)
covariate = np.random.randn(n_samples, 2)  # 2 covariates

# Initialize regressor
regressor = CovariateRegressor(covariate=covariate, X_full=X, estimator=LinearRegression())

# Fit and transform
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
regressor.fit(X_train)
X_residuals = regressor.transform(X_test)

# Check output shape
assert X_residuals.shape == X_test.shape

# Check that weights were fitted
assert regressor.weights_ is not None
assert len(regressor.weights_[0]) == 3  # 2 covariates + intercept
assert len(regressor.weights_) == n_features

"""
Test CovariateRegressor with cross_validate=False.

Scenario: Disable cross-validation to test the regressor with
standard non-cross-validated fitting.
"""
np.random.seed(42)
n_samples, n_features = 50, 10

X = np.random.randn(n_samples, n_features)
covariate = np.random.randn(n_samples, 1)

regressor = CovariateRegressor(
    covariate=covariate, X_full=X, cross_validate=False
)

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
regressor.fit(X_train)
X_residuals = regressor.transform(X_test)

assert X_residuals.shape == X_test.shape

# %%
