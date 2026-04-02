
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

#I think these need to be set here for some reaosn
con_filepath = '/gscratch/scrubbed/gkolpin/xcpd_output/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd'
con_model = 'lassoBIC_blocks'
p_filepath = '/gscratch/scrubbed/gkolpin/phenotype_data/study-PNC_desc-participants.tsv'
scaler = StandardScaler() #making it normal so things work with it
sublist_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/rand709_sub_list.txt'
random_state = 10

#and need the sublist
sublist = []
with open(sublist_filepath, 'r') as file:
    sublist = file.read().splitlines()
print(len(sublist))

#and network labels
atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_filename = atlas.maps
atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas17_filename = atlas.maps

full_labels = atlas.labels[1:]
labels_7 = [label.split("_")[2] for label in full_labels]

full_labels_17 = atlas_17.labels[1:]
labels_17 = [label.split("_")[2] for label in full_labels_17]


def get_connectomes(con_model, network=False, con_filepath=con_filepath, sublist=sublist, network_labels=labels_7):
    connectomes = {}
    timeseries_pred = {}
    for sub in sublist:
        for file in glob.glob(op.join(con_filepath,
                                 con_model,
                                 f'sub-{sub}',
                                 '*results.pkl')):
            with open(file, 'rb') as i:
                loaded_data = pkl.load(i)
                if con_model == 'correlation_random':
                    connectome = pd.DataFrame(loaded_data['fc_matrix'])
                else:
                    connectome = pd.DataFrame(loaded_data['fold_0']['fc_matrix'])
                    timeseries_pred[sub] = np.mean([loaded_data['fold_0'][f'node_{node}']['test_r2'] for node in range(100)])
                if network:
                        connectome['network'] = network_labels
                        connectome = connectome.groupby('network', as_index=False).mean()
                        connectome.drop('network', axis=1, inplace=True)
                        connectome = connectome.T
                        connectome['network'] = network_labels
                        connectome = connectome.groupby('network', as_index=False).mean()
                        connectome.drop('network', axis=1, inplace=True)
                        connectome = connectome.T
                connectome = connectome.to_numpy()                    
                if not network:
                    np.fill_diagonal(connectome, 0)
                    if con_model == 'correlation_random':
                        connectome = np.tril(connectome, k=-1)
                connectome = connectome.ravel()
                connectomes[sub] = connectome

    connectomes = pd.DataFrame(connectomes).transpose()
    if network:
        num_networks = len(set(network_labels))
        net_names = []
        seen = set()
        for item in network_labels:
            if item not in seen:
                net_names.append(item)
                seen.add(item)
        names = [f"{x}_to_{y}" for x in net_names for y in net_names]
        connectomes.rename(columns={num: names[num] for num in range(num_networks**2)}, inplace=True)
    else:
        connectomes.rename(columns={num: f'edge_{num}' for num in range(10000)}, inplace=True)
    connectomes.index = np.int64(connectomes.index)
    connectomes.index.name = 'participant_id'
    if con_model == 'correlation_random':
        return connectomes
    else:
        return connectomes, timeseries_pred


def cpm(X, y, folds, correlate, alpha, model_obj, corr_matricies, plot=False):
    '''
    X; connectomes (only makes sense to use the full connectomes)
    y; predictor
    folds; folds for total fitting
    correlate; what method of correlating edges to y (only basic corr now, but want to maybe add partial corr)
    alpha; p cuttoff for selecting edges to include (I feel like it should be corrected for because we're doing so many tests)
    model; prediciton model (they just do linear regression, but no reason it couldn't be ridge)
    '''
    results = {}
    predictions = []
    edge_labels = X.columns
    y = np.array(y)
    kf = KFold(folds)
    fold = 0
    k_fold_y = []
    if plot:
        f, asx = plt.subplots(folds, None)
    for train_index, test_index in kf.split(X):
        r_vals = []
        p_vals = []
        pos_net = set()
        neg_net = set()
        sub_sums = pd.DataFrame(index=X.index, columns=['positive', 'negative', 'total'])
    
        train_X = X.iloc[train_index]
        train_y = y[train_index]
        test_X = X.iloc[test_index]
        test_y = y[test_index]
    
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
        pos_sum = X[pos_cols].sum(axis=1)
        neg_sum = X[neg_cols].sum(axis=1)
        total = abs(pos_sum) + abs(neg_sum)

        if corr_matricies:
            sub_sums['positive'] = pos_sum.values / 2
            sub_sums['negative'] = neg_sum.values / 2
            sub_sums['total'] = total.values / 2
        else:
            sub_sums['positive'] = pos_sum.values
            sub_sums['negative'] = neg_sum.values
            sub_sums['total'] = total.values
        train_sub_sums = sub_sums.iloc[train_index]
        test_sub_sums = sub_sums.iloc[test_index]
        #could theoretically call my model fitting function here, if I make it work with 0 folds (could also additionally crossvalidate this step?)
        pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
        pl.fit(train_sub_sums[['total']], train_y)
        results[f'fold {fold} stats'] = {'test': pl.score(test_sub_sums[['total']], test_y), 'train': pl.score(train_sub_sums[['total']], train_y)}
        predictions.extend(pl.predict(test_sub_sums[['total']]))
        #can refit and repredict for pos and neg networks, if either is better I would be very confused.
        k_fold_y.extend(test_y)
        print(f'fold {fold} complete')
        fold += 1
    
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['MAE'] = mean_absolute_error(k_fold_y, predictions)
    results['predictions'] = predictions
    results['r'] = np.corrcoef(k_fold_y, predictions)
    results['residuals'] = [y - yprime for y, yprime in zip(results['k_fold_age'], results['predictions'])]

    return(results)


def fit_model(X, y, model_obj, folds=5, component=None, verbose=False):
    '''
    TO DO
    '''
    results = {}
    predictions = []
    groups = np.array(X.index)
    X = np.array(X)
    y = np.array(y)
    if component == 'PCA':
        pl = Pipeline([('scaler', StandardScaler()), ('PCA', PCA()), ('model', model_obj)])
    elif component == 'ICA':
        pl = Pipeline([('scaler', StandardScaler()), ('ICA', FastICA()), ('model', model_obj)])
    else:
        pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
    group_kf = GroupKFold(folds)
    k_fold_y = []
    for fold, (train_index, test_index) in enumerate(group_kf.split(X, y, groups)):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]

        pl.fit(train_X, train_y)      
        results[f'fold {fold} stats'] = [{'test': pl.score(test_X, test_y), 'train': pl.score(train_X, train_y)}]
        results[f'fold {fold} coef'] = model_obj.coef_
        
        predictions.extend(pl.predict(test_X))
        k_fold_y.extend(test_y)
        if verbose:
            print(f'fold {fold} complete')
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['MAE'] = mean_absolute_error(k_fold_y, predictions)
    results['predictions'] = predictions
    results['r'] = np.corrcoef(k_fold_y, predictions)[0, 1]
    results['residuals'] = [y - yprime for y, yprime in zip(results['k_fold_age'], results['predictions'])]
    return(results)