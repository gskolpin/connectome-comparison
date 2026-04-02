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
def get_connectomes(con_model, network=False, con_filepath=con_filepath, sublist=sublist, network_labels=labels_7):
    connectomes = {}
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
                if network:
                        connectome['network'] = network_labels
                        connectome = connectome.groupby('network', as_index=False).mean()
                        connectome.drop('network', axis=1, inplace=True)
                        connectome = connectome.T
                        connectome['network'] = network_labels
                        connectome = connectome.groupby('network', as_index=False).mean()
                        #print(connectome)
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
    return(connectomes)


# %%
net7_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_7)
net17_lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks', network=True, network_labels=labels_17)

# %%
net7_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_7)
net17_corr_connectomes = get_connectomes(con_model='correlation_random', network=True, network_labels=labels_17)
#net17_corr_connectomes.head()

# %%
corr_connectomes = get_connectomes(con_model='correlation_random')
#corr_connectomes.head()
print(len(corr_connectomes))

# %%
lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks')
#lasso_connectomes.head()
print(len(lasso_connectomes))

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
'''
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
print(results['full_r2'])


# %%
def cpm(X, y, folds, correlate, alpha, model_obj, corr_matricies, plot=False):
    '''
    X; connectomes (only makes sense to use the full connectomes)
    y; predictor
    folds; folds for total fitting
    correlate; what method of correlating edges to y (only basic corr now, but want to maybe add partial corr)
    alpha; p cuttoff for selecting edges to include (I feel like it should be corrected for because we're doing so many tests, but they don't mention that in the shen paper)
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
        pl.fit(train_sub_sums[['negative']], train_y)
        results[f'fold {fold} stats'] = {'test': pl.score(test_sub_sums[['negative']], test_y), 'train': pl.score(train_sub_sums[['negative']], train_y)}
        predictions.extend(pl.predict(test_sub_sums[['negative']]))
        #can refit and repredict for pos and neg networks, if either is better I would be very confused.
        k_fold_y.extend(test_y)
        print(f'fold {fold} complete')
        fold += 1
    
    results['k_fold_y'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['predictions'] = predictions

    return(results)


# %%
lasso_data = lasso_data.dropna(axis=0, subset=[p])
corr_data = corr_data.dropna(axis=0, subset=[p])
net7_corr_data = net7_corr_data.dropna(axis=0, subset=[p])
net7_lasso_data = net7_lasso_data.dropna(axis=0, subset=[p])
net17_corr_data = net17_corr_data.dropna(axis=0, subset=[p])
net17_lasso_data = net17_lasso_data.dropna(axis=0, subset=[p])

p = 'p_factor_mcelroy_harmonized_all_samples'
cpm_corr_results = cpm(corr_data[corr_data.columns[:10000]], corr_data[p], 10, 'corr', .01, LinearRegression(), True)
cpm_lasso_results = cpm(lasso_data[corr_data.columns[:10000]], lasso_data[p], 10, 'corr', .01, LinearRegression(), False)
print(cpm_corr_results['full_r2'])
print(cpm_lasso_results['full_r2'])

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
"just age with p factor plotted"
plt.scatter(lasso_data['age'], lasso_data[p])
lasso_b, lasso_a = np.polyfit(lasso_data['age'], lasso_data[p], 1)
plt.plot(lasso_data['age'], lasso_a + lasso_b * lasso_data['age'], color='red')

# %%
#I don't know if doing CPM on the network connectomes makes sense, usually people do the grouping afterwards from the positive and negative networks.
#how different are the sum values of people?
#would just negative work better? there are more of them, is that typical generally?

# %%
#its okay now
#print(type(rand100_phenotype.index[1]))
#print(type(connectomes.index[1]))
#for index in connectomes.index:
#    assert index in rand100_phenotype.index

# %%
'''
does not work
X = net7_corr_data[net7_corr_data.columns[:49]]
y = net7_corr_data['age']
fracs = np.linspace(0, 1, n_alphas)
fracridge(X, y, fracs=fracs)
'''

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
    age_matrix.append(scipy.stats.pearsonr(data[edge], data[p])[0])
age_matrix.reverse()
age_matrix = np.reshape(age_matrix, (7, 7))
#age_matrix = np.tril(age_matrix)

sns.heatmap(age_matrix, annot=True, cmap='bwr', fmt='.2f',ax=axs[0], linewidths=0.5, vmin=-0.25, vmax=0.25) 
axs[0].set_xticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_yticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_title('Network Connectivity Correlations to Age')


axs[1].scatter(data[p], data[vis_edge], alpha=0.5)
n7_b, n7_a = np.polyfit(data['age'], data[vis_edge], 1)
axs[1].plot(data[p], n7_a + n7_b * data[p], color='darkorange')
axs[1].set_xlabel('p factor')
axs[1].set_ylabel('LASSO Functional Connectivity (r)', rotation=90)
axs[1].set_title('Limbic - Limbic')
fig.tight_layout()
plt.show()
#ax.set_yticks(locations)
#ax.set_yticks(locations)
#ax.set_xticklabels(net7_labels)
#ax.set_yticklabels(net7_labels)

# %%
#for data merging, move things out, be clearer, erorrs can happen there
folds = 10
#model = LassoCV(random_state=random_state)
#model = LinearRegression()
#fracs = np.linspace(0, 1, n_alphas)
#model = FracRidgeRegressor(fracs=fracs, fit_intercept=True) #fits 20 predictions, one per alpha, sooooo would need to manually select the best alpha?
model = RidgeCV()
lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data[p], model, folds=folds)
corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data[p], model, folds=folds)
net7_lasso_results = fit_model(net7_lasso_data[net7_lasso_connectomes.columns], net7_lasso_data[p], model, folds=folds)
net17_lasso_results = fit_model(net17_lasso_data[net17_lasso_connectomes.columns], net17_lasso_data[p], model, folds=folds)
net7_corr_results = fit_model(net7_corr_data[net7_corr_connectomes.columns], net7_corr_data[p], model, folds=folds)
net17_corr_results = fit_model(net17_corr_data[net17_corr_connectomes.columns], net17_corr_data[p], model, folds=folds)

# %%
print('lasso:', lasso_results['fold 1 stats'], 'full r2:', lasso_results['full_r2']) #looks just like a lot of overfitting
print('correlation:', corr_results['fold 1 stats'], 'full r2:', corr_results['full_r2']) #um huh, maybe just underfitting from not enough subjects??? weird b/c still lots of features
print('lasso 7 networks:', net7_lasso_results['fold 1 stats'], 'full r2:', net7_lasso_results['full_r2'])
print('lasso 17 networks:', net17_lasso_results['fold 1 stats'], 'full r2:', net17_lasso_results['full_r2'])
print('correlation 7 networks:', net7_corr_results['fold 1 stats'], 'full r2:', net7_corr_results['full_r2'])
print('correlation 17 networks:', net17_corr_results['fold 1 stats'], 'full r2:', net17_corr_results['full_r2'])

# %%
#for data merging, move things out, be clearer, erorrs can happen there
folds = 10
model = LassoCV(random_state=random_state)
#model = LinearRegression()
#fracs = np.linspace(0, 1, n_alphas)
#model = FracRidgeRegressor(fracs=fracs, fit_intercept=True) #fits 20 predictions, one per alpha, sooooo would need to manually select the best alpha?
#model = RidgeCV()
pca_lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data[p], model, folds=folds, component='PCA')
#pca_corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], model, folds=folds, component='PCA')

# %%
plt.scatter(pca_lasso_results['k_fold_age'], pca_lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(pca_lasso_results['k_fold_age'], pca_lasso_results['predictions'], 1)
plt.plot(pca_lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data[p], color='red')

plt.title('PCA lasso age prediction')
plt.show()

# %%
#plotting residuals
plt.scatter(lasso_data['age'], pca_lasso_results['residuals'])

# %%
print('lasso:', pca_lasso_results['fold 1 stats'], 'full r2:', pca_lasso_results['full_r2']) #looks just like a lot of overfitting
#print('correlation:', pca_corr_results['fold 1 stats'], 'full r2:', pca_corr_results['full_r2'])

# %%
#TO DO: correlate residuals with p values??? at least plot them... if I see outliers like their 10% i could remove those

# %%
#TO DO: check age stratified (maybe like ranging 2-5 buckets?)

# %%
#TO DO: add PClasso (and grouplasso (maybe with PCA), and fracridge) to the models and comparisons (really should be up there but down here to remember)

# %%
#TO DO: get the nilearn correlaiton data visualizations working??? (the cai figure 3) (this is definetly low on priorities rn though)

# %%
f, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 16))
axs[0, 0].scatter(corr_results['k_fold_age'], corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(corr_results['k_fold_age'], corr_results['predictions'], 1)
axs[0, 0].plot(corr_results['k_fold_age'], corr_a + corr_b * corr_data[p], color='red')
corr00 = np.corrcoef(corr_results['k_fold_age'], corr_results['predictions'])[1, 0]
axs[0, 0].annotate(f'$r$={corr00:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[0, 1].scatter(lasso_results['k_fold_age'], lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(lasso_results['k_fold_age'], lasso_results['predictions'], 1)
axs[0, 1].plot(lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data[p], color='red')
corr01 = np.corrcoef(lasso_results['k_fold_age'], lasso_results['predictions'])[1, 0]
axs[0, 1].annotate(f'$r$={corr01:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[1, 0].scatter(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], 1)
axs[1, 0].plot(net7_corr_results['k_fold_age'], n7_a + n7_b * net7_corr_data[p], color='red')
corr10 = np.corrcoef(net7_corr_results['k_fold_age'], net7_corr_results['predictions'])[1, 0]
axs[1, 0].annotate(f'$r$={corr10:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[1, 1].scatter(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], 1)
axs[1, 1].plot(net7_lasso_results['k_fold_age'], n7_a + n7_b * net7_lasso_data[p], color='red')
corr11 = np.corrcoef(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'])[1, 0]
axs[1, 1].annotate(f'$r$={corr11:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[2, 0].scatter(net17_corr_results['k_fold_age'],net17_corr_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_corr_results['k_fold_age'], net17_corr_results['predictions'], 1)
axs[2, 0].plot(net17_corr_results['k_fold_age'], n17_a + n17_b * net17_corr_data[p], color='red')
corr20 = np.corrcoef(net17_corr_results['k_fold_age'], net17_corr_results['predictions'])[1, 0]
axs[2, 0].annotate(f'$r$={corr20:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

axs[2, 1].scatter(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], 1)
axs[2, 1].plot(net17_lasso_results['k_fold_age'], n17_a + n17_b * net17_lasso_data[p], color='red')
corr21 = np.corrcoef(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'])[1, 0]
axs[2, 1].annotate(f'$r$={corr21:.3f}', xy=(0.7, 0.9), xycoords='axes fraction')

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
print('min:',  rand100_phenotype['age'].min(), 'max:', rand100_phenotype['age'].max())

# %%
lasso_data.head()

# %%
#predict from age??? sanity check
check_results = fit_model(corr_data[['age']], corr_data[p], LinearRegression(), folds=folds)
print('check:', check_results['fold 1 stats'], 'full r2:', check_results['r'][0, 1])

# %%
#okay now covariate regressor.
import numpy as np
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def find_subset_indices(X_full, X_subset, method="hash", allow_missing=False):
    """
    Find row indices in X_full that correspond to rows in X_subset.
    Supports 'hash' (fast) and 'precise' (element-wise) matching.
    Allow_missing appends empty array for non-matching rows if True.
    """
    if X_full.shape[1] != X_subset.shape[1]:
        raise ValueError(
            f"Feature dimensions don't match: {X_full.shape[1]} vs {X_subset.shape[1]}"
        )
    indices = []
    if method == "precise":
        for i, subset_row in enumerate(X_subset):
            matches = [
                j
                for j, full_row in enumerate(X_full)
                if np.array_equal(full_row, subset_row, equal_nan=True)
            ]
            if not matches and not allow_missing:
                raise ValueError(f"No matching row found for subset row {i}")
            indices.append(matches[0] if matches else [])
    elif method == "hash":
        full_hashes = [hash(row.tobytes()) for row in X_full]
        for i, subset_row in enumerate(X_subset):
            subset_hash = hash(subset_row.tobytes())
            try:
                indices.append(full_hashes.index(subset_hash))
            except ValueError as e:
                if allow_missing:
                    indices.append([])
                else:
                    raise ValueError(f"No matching row found for subset row {i}") from e
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hash' or 'precise'.")
    return np.array(indices)



class CovariateRegressor(BaseEstimator, TransformerMixin):
    """
    Fits covariate(s) onto each feature in X and returns their residuals.
    """

    def __init__(
        self,
        covariate,
        X_full,
        pipeline=None,
        cross_validate=True,
        precise=False,
        unique_id_col_index=None,
        stack_intercept=True,
    ):
        """Regresses out a variable (covariate) from each feature in X.

        Parameters
        ----------
        covariate : numpy array
            Array of length (n_samples, n_covariates) to regress out of each
            feature; May have multiple columns for multiple covariates.
        X_full : numpy array
            Array of length (n_samples, n_features), from which the covariate
            will be regressed. This is used to determine how the
            covariate-models should be cross-validated (which is necessary
            to use in in scikit-learn Pipelines).
        pipeline : sklearn.pipeline.Pipeline or None, default=None
            Optional scikit-learn pipeline to apply to the covariate before fitting
            the regression model. If provided, the pipeline will be fitted on the
            covariate data during the fit phase and applied to transform the covariate
            in both fit and transform phases. This allows for preprocessing steps
            such as imputation, scaling, normalization, or feature engineering to be
            applied to the covariate consistently across train and test sets. If None,
            the covariate is used as-is without any preprocessing.
        cross_validate : bool
            Whether to cross-validate the covariate-parameters (y~covariate)
            estimated from the train-set to the test set (cross_validate=True)
            or whether to fit the covariate regressor separately on the test-set
            (cross_validate=False).
        precise: bool
            When setting precise to True, the arrays are compared feature-wise,
            which is accurate, but relatively slow. When setting precise to False,
            it will infer the index of the covariates by looking at the hash of all
            the features, which is much faster. Also, to aid the accuracy, we remove
            the features which are constant (0) across samples.
        stack_intercept : bool
            Whether to stack an intercept to the covariate (default is True)

        Attributes
        ----------
        weights_ : numpy array
            Array with weights for the covariate(s).

        Notes
        -----
        This is a modified version of the ConfoundRegressor from [1]_. Setting
        cross_validate to True is equivalent to "foldwise covariate regression" (FwCR)
        as described in Snoek et al. (2019). Setting this parameter to False, however,
        is NOT equivalent to "whole dataset covariate regression" (WDCR) as it does not
        apply covariate regression to the *full* dataset, but simply refits the
        covariate model on the test-set. We recommend setting this parameter to True.
        Transformer-objects in scikit-learn only allow to pass the data (X) and
        optionally the target (y) to the fit and transform methods. However, we need
        to index the covariate accordingly as well. To do so, we compare the X during
        initialization (self.X_full) with the X passed to fit/transform. As such, we can
        infer which samples are passed to the methods and index the covariate
        accordingly. The precise flag controls the precision of the index matching.

        References
        ----------
        .. [1] Lukas Snoek, Steven Miletić, H. Steven Scholte,
            "How to control for confounds in decoding analyses of neuroimaging data",
            NeuroImage, Volume 184, 2019, Pages 741-760, ISSN 1053-8119,
            https://doi.org/10.1016/j.neuroimage.2018.09.074.
        """
        self.covariate = covariate.astype(np.float64)
        self.cross_validate = cross_validate
        self.X_full = X_full
        self.precise = precise
        self.stack_intercept = stack_intercept
        self.weights_ = None
        self.pipeline = pipeline
        self.imputer = SimpleImputer(strategy="median")
        self.X_imputer = SimpleImputer(strategy="median")
        self.unique_id_col_index = unique_id_col_index

    def _prepare_covariate(self, covariate):
        """Prepare covariate matrix (adds intercept if needed)"""
        if self.stack_intercept:
            return np.c_[np.ones((covariate.shape[0], 1)), covariate]
        return covariate

    def fit(self, X, y=None):
        """Fits the covariate-regressor to X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!
        y : None
            Included for compatibility; does nothing.
        """

        # Prepare covariate matrix (adds intercept if needed)
        covariate = self._prepare_covariate(self.covariate)

        # Find indices of X subset in the original X
        method = "precise" if self.precise else "hash"
        fit_idx = find_subset_indices(self.X_full, X, method=method)

        # Remove unique ID column if specified
        if self.unique_id_col_index is not None:
            X = np.delete(X, self.unique_id_col_index, axis=1)

        # Extract covariate data for the fitting subset
        covariate_fit = covariate[fit_idx, :]

        # Conditional imputation for covariate data
        if np.isnan(covariate_fit).any():
            covariate_fit = self.imputer.fit_transform(covariate_fit)
        else:
            # Still fit the imputer for consistency in transform
            self.imputer.fit(covariate_fit)

        # Apply pipeline transformation if specified
        if self.pipeline is not None:
            X = self.pipeline.fit_transform(X)

        # Conditional imputation for X
        if np.isnan(X).any():
            X = self.X_imputer.fit_transform(X)
        else:
            # Still fit the imputer for consistency in transform
            self.X_imputer.fit(X)

        # Fit linear regression: X = covariate * weights + residuals
        # Using scipy's lstsq for numerical stability
        self.weights_ = lstsq(covariate_fit, X)[0] #CHANGHE HERE TO FIT #########################################################################

        return self

    def transform(self, X):
        """Regresses out covariate from X.

        Parameters
        ----------
        X : numpy array
            An array of shape (n_samples, n_features), which should correspond
            to your train-set only!

        Returns
        -------
        X_new : ndarray
            ndarray with covariate-regressed features
        """

        if not self.cross_validate:
            self.fit(X)

        # Prepare covariate matrix (adds intercept if needed)
        covariate = self._prepare_covariate(self.covariate)

        # Find indices of X subset in the original X
        method = "precise" if self.precise else "hash"
        transform_idx = find_subset_indices(self.X_full, X, method=method)

        # Remove unique ID column if specified
        if self.unique_id_col_index is not None:
            X = np.delete(X, self.unique_id_col_index, axis=1)

        # Extract covariate data for the transform subset
        covariate_transform = covariate[transform_idx]

        # Conditional imputation for covariate data (use fitted imputer)
        if np.isnan(covariate_transform).any():
            covariate_transform = self.imputer.transform(covariate_transform)

        # Apply pipeline transformation if specified
        if self.pipeline is not None:
            X = self.pipeline.transform(X)

        # Conditional imputation for X (use fitted imputer)
        if np.isnan(X).any():
            X = self.X_imputer.transform(X)

        # Compute residuals
        X_new = X - covariate_transform.dot(self.weights_) # CHANGE HERE TO TRUE - PREDICTED #################################################

        # Ensure no NaNs in output
        X_new = np.nan_to_num(X_new)

        return X_new


# %%
lasso_data = lasso_data.dropna(axis=0, subset=[p])
corr_data = corr_data.dropna(axis=0, subset=[p])

X_full_lasso = lasso_data[[f'edge_{num}' for num in range(10000)]]
X_full_lasso = X_full_lasso.to_numpy(dtype=float)
lasso_covariate = lasso_data['age']
lasso_predictor = lasso_data[p]

X_full_corr = corr_data[[f'edge_{num}' for num in range(10000)]]
X_full_corr = X_full_corr.to_numpy(dtype=float)
corr_covariate = corr_data['age']
corr_predictor = corr_data[p]

# %%
#actual covariate regressor
lasso_cov_reg = CovariateRegressor(
    covariate=lasso_covariate,
    X_full=X_full_lasso,
    cross_validate=True
)

corr_cov_reg = CovariateRegressor(
    covariate=corr_covariate,
    X_full=X_full_corr,
    cross_validate=True
)

X_residual_lasso = lasso_cov_reg.fit_transform(X_full_lasso)
X_residual_corr = corr_cov_reg.fit_transform(X_full_corr)

# %%
covreg_lasso_results = fit_model(X_residual_lasso, lasso_predictor, LassoCV(), folds=folds)
covreg_corr_results = fit_model(X_residual_corr, corr_predictor, LinearRegression(), folds=folds)

# %%
print('corr: r,', covreg_corr_results['r'], 'r2', covreg_corr_results['full_r2'])
print('lasso: r,', covreg_lasso_results['r'], 'r2', covreg_lasso_results['full_r2'])

# %%
print(len(corr_results['predictions']))

# %%
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(10, 5))
ax1.scatter(covreg_corr_results['k_fold_age'], covreg_corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(covreg_corr_results['k_fold_age'], covreg_corr_results['predictions'], 1)
ax1.plot(covreg_corr_results['k_fold_age'], corr_a + corr_b * corr_data[p], color='red')

ax2.scatter(covreg_lasso_results['k_fold_age'], covreg_lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(covreg_lasso_results['k_fold_age'], covreg_lasso_results['predictions'], 1)
ax2.plot(covreg_lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data[p], color='red')

ax1.set_title('covreg p correlation')
ax2.set_title('covreg p lasso')
plt.show()

# %%
