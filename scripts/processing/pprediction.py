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

from fracridge import fracridge, FracRidgeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

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
                if con_model == 'lassoBIC_blocks':
                    connectome = pd.DataFrame(loaded_data['fold_0']['fc_matrix'])
                if con_model == 'correlation_random':
                    connectome = pd.DataFrame(loaded_data['fc_matrix'])
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
print(phenotype.index)

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
#TO DO: try and get CPM working
def cpm(X, y, folds, correlate, alpha, model_obj, corr_matricies):
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
    for train_index, test_index in kf.split(X):
        r_vals = []
        p_vals = []
        pos_net = set()
        neg_net = set()
        sub_sums = pd.DataFrame(index=X.index, columns=['positive', 'negative', 'total'])
        fold += 1
    
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
        total = pos_sum + neg_sum
        
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
        k_fold_y.extend(test_y)
        print(f'fold {fold} complete')
    
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['predictions'] = predictions
    return(results)


# %%
cpm_corr_results = cpm(corr_data[corr_data.columns[:10000]], corr_data['age'], 5, 'corr', .01, LinearRegression(), True)
cpm_lasso_results = cpm(lasso_data[corr_data.columns[:10000]], lasso_data['age'], 5, 'corr', .01, LinearRegression(), False)
print(cpm_corr_results['full_r2'])
print(cpm_lasso_results['full_r2'])

# %%
#I don't know if doing CPM on the network connectomes makes sense, usually people do the grouping afterwards from the positive and negative networks.

# %%
#more testing somewhere else because that took a long time
train_sub_sums = sub_sums.iloc[train_index]
print(len(train_sub_sums))


# %%
def fit_model(X, y, model_obj, folds=5):
    '''
    TO DO
    '''
    results = {}
    predictions = []
    X = np.array(X)
    y = np.array(y)
    pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
    kf = KFold(folds)
    fold = 0
    k_fold_y = []
    for train_index, test_index in kf.split(X):
        fold += 1
        #print(f" Train: shape={np.shape(train_index)}")
        #print(f" Test: shape={np.shape(test_index)}")
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]

        pl.fit(train_X, train_y).score(test_X, test_y)            
        results[f'fold {fold} stats'] = {'test': pl.score(test_X, test_y), 'train': pl.score(train_X, train_y)}
        results[f'fold {fold} coef'] = model_obj.coef_
        results[f'fold {fold} alpha'] = model_obj.alpha_
        
        #can change both of these to maybe be one df that has the id's again as index? maybe just other columnn the way Ariel reccomended
        predictions.extend(pl.predict(test_X))
        k_fold_y.extend(test_y)
        print(f'fold {fold} complete')
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['predictions'] = predictions
    return(results)


# %%
#its okay now
#print(type(rand100_phenotype.index[1]))
#print(type(connectomes.index[1]))
#for index in connectomes.index:
#    assert index in rand100_phenotype.index

# %%
n_alphas = 20
fracs = np.linspace(0, 1, n_alphas)
model = RidgeCV()
lasso_results = model.fit(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'])
print(len(lasso_data['age']))
print(len(model.predict(lasso_data[[f'edge_{num}' for num in range(10000)]])))

# %%
#for data merging, move things out, be clearer, erorrs can happen there
folds = 5
#model = LassoCV(random_state=random_state)
#model = LinearRegression()
#fracs = np.linspace(0, 1, n_alphas)
#model = FracRidgeRegressor(fracs=fracs, fit_intercept=True) #fits 20 predictions, one per alpha, sooooo would need to manually select the best alpha?
model = RidgeCV()
lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], model, folds=folds)
corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], model, folds=folds)
net7_lasso_results = fit_model(net7_lasso_data[net7_lasso_connectomes.columns], net7_lasso_data['age'], model, folds=folds)
net17_lasso_results = fit_model(net17_lasso_data[net17_lasso_connectomes.columns], net17_lasso_data['age'], model, folds=folds)
net7_corr_results = fit_model(net7_corr_data[net7_corr_connectomes.columns], net7_corr_data['age'], model, folds=folds)
net17_corr_results = fit_model(net17_corr_data[net17_corr_connectomes.columns], net17_corr_data['age'], model, folds=folds)

# %%
#Okay soo this can get the coefficient path plots from seemingly where they start changing to all 0
alphas = np.logspace(-4, 10, 50)
coefs = []
for a in alphas:
    #model = Lasso(random_state=random_state, alpha=a)
    model = Ridge(alpha = a)
    model.fit(net7_corr_data[net7_corr_data.columns[:49]], net7_corr_data['age']) #looks fairly normal
    #model.fit(net7_lasso_data[net7_corr_data.columns[:49]], net7_lasso_data['age']) #super huge coefficients
    #model.fit(net17_corr_data[net17_corr_data.columns[:289]], net17_corr_data['age'])
    #model.fit(net17_lasso_data[net17_corr_data.columns[:289]], net17_lasso_data['age']) #can lower to -7 for better view, but just bad overal
    #model.fit(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'])
    #model.fit(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'])
    coefs.append(model.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Lasso Coefficients vs Regularization Strength (alpha)") #change
plt.axis("tight")
plt.show()


# %%
#okay now lasso, with all of these, just seems to be not at all working, so lets try fracridge!
X = net7_lasso_data[net7_lasso_data.columns[:49]]
y = net7_lasso_data['age']
n_alphas = 20
alphas = np.logspace(-10, 10, n_alphas)
rr_alphas = np.logspace(-10, 10, n_alphas)
rr_coefs = []
rr_coefs = np.zeros((X.shape[-1], n_alphas))
rr_pred = np.zeros((y.shape[-1], n_alphas))

#just ridge for comparison
for aa in range(len(rr_alphas)):
    RR = Ridge(alpha=rr_alphas[aa], fit_intercept=True)
    RR.fit(X, y)
    rr_coefs[:, aa] = RR.coef_
    rr_pred[:, aa] = cross_val_predict(RR, X, y)

#fracridge
fracs = np.linspace(0, 1, n_alphas)
FR = FracRidgeRegressor(fracs=fracs, fit_intercept=True)
FR.fit(X, y)
fr_pred = cross_val_predict(FR, X, y)

#and plot
fig, ax = plt.subplots(1, 2)
ax[0].plot(fracs, FR.coef_.T)
ylims = ax[0].get_ylim()
ax[0].vlines(fracs, ylims[0], ylims[1], linewidth=0.5, color='gray')
ax[0].set_ylim(*ylims)
ax[1].plot(np.log(rr_alphas[::-1]), rr_coefs.T)
ylims = ax[1].get_ylim()
ax[1].vlines(np.log(rr_alphas[::-1]), ylims[0], ylims[1], linewidth=0.5,
             color='gray')
ax[1].set_ylim(*ylims)


# %%
'''
does not work
X = net7_corr_data[net7_corr_data.columns[:49]]
y = net7_corr_data['age']
fracs = np.linspace(0, 1, n_alphas)
fracridge(X, y, fracs=fracs)
'''

# %%
print('lasso:', lasso_results['fold 1 stats'], 'full r2:', lasso_results['full_r2']) #looks just like a lot of overfitting
print('correlation:', corr_results['fold 1 stats'], 'full r2:', corr_results['full_r2']) #um huh, maybe just underfitting from not enough subjects??? weird b/c still lots of features
print('lasso 7 networks:', net7_lasso_results['fold 1 stats'], 'full r2:', net7_lasso_results['full_r2'])
print('lasso 17 networks:', net17_lasso_results['fold 1 stats'], 'full r2:', net17_lasso_results['full_r2'])
print('correlation 7 networks:', net7_corr_results['fold 1 stats'], 'full r2:', net7_corr_results['full_r2'])
print('correlation 17 networks:', net17_corr_results['fold 1 stats'], 'full r2:', net17_corr_results['full_r2'])

# %%
# make some new ones for comparison
#fully crossvalidated
print('lasso: full r2:', lasso_results['full_r2']) #looks just like a lot of overfitting
print('correlation: full r2:', corr_results['full_r2']) #um huh, maybe just underfitting from not enough subjects??? weird b/c still lots of features
print('lasso 7 networks: full r2:', net7_lasso_results['full_r2'])
print('lasso 17 networks: full r2:', net17_lasso_results['full_r2'])
print('correlation 7 networks: full r2:', net7_corr_results['full_r2'])
print('correlation 17 networks: full r2:', net17_corr_results['full_r2'])

# %%
print(net7_corr_results['fold 1 coef'])
print(net7_corr_results['fold 1 alpha'])

# %%
data = net7_lasso_data
#Recreating the RBC figure 7 stuff
vis_edge = 'SalVentAttn_to_SalVentAttn'
vis_dfn_edge = 'SalVentAttn_to_Default'
net7_labels = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Control', 'Default Mode']
locations = np.arange(len(net7_labels))
#first just the matrix, want to see
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
age_matrix = []
for edge in net7_corr_data.columns[:49]:
    age_matrix.append(scipy.stats.pearsonr(data[edge], data['age'])[0])
age_matrix = np.reshape(age_matrix, (7, 7))
#age_matrix = np.tril(age_matrix)

sns.heatmap(age_matrix, annot=True, cmap='bwr', fmt='.2f',ax=axs[0], linewidths=0.5, vmin=-0.25, vmax=0.25) 
axs[0].set_xticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')
axs[0].set_yticks(locations, net7_labels, rotation=45, ha='right', rotation_mode='anchor')

axs[1].scatter(data['age'], data[vis_edge], alpha=0.5)
n7_b, n7_a = np.polyfit(data['age'], data[vis_edge], 1)
axs[1].plot(data['age'], n7_a + n7_b * data['age'], color='red')
axs[1].set_xlabel('age (years)')
axs[1].set_ylabel('Functional Connectivity', rotation=90)
axs[1].set_title('Ventral Attention - Ventral Attention')
fig.tight_layout()
plt.show()
#ax.set_yticks(locations)
#ax.set_yticks(locations)
#ax.set_xticklabels(net7_labels)
#ax.set_yticklabels(net7_labels)

# %%
#TO DO: check age stratified (maybe like ranging 2-5 buckets?)

# %%
#TO DO: add ridge and PClasso (and grouplasso (maybe with PCA), and fracridge) to the models and comparisons (really should be up there but down here to remember)

# %%
#TO DO: get the nilearn correlaiton data visualizations working??? (the cai figure 3) (this is definetly low on priorities rn though)

# %%
#want to visualize some stuff, but can do that quickly maybe:    
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(corr_results['k_fold_age'], corr_results['predictions'], alpha=0.5)
corr_b, corr_a = np.polyfit(corr_results['k_fold_age'], corr_results['predictions'], 1)
ax1.plot(corr_results['k_fold_age'], corr_a + corr_b * corr_data['age'], color='red')

ax2.scatter(lasso_results['k_fold_age'], lasso_results['predictions'], alpha=0.5)
lasso_b, lasso_a = np.polyfit(lasso_results['k_fold_age'], lasso_results['predictions'], 1)
ax2.plot(lasso_results['k_fold_age'], lasso_a + lasso_b * lasso_data['age'], color='red')

ax1.set_title('correlation')
ax2.set_title('lasso')
plt.show()

# %%
#same as above but for the networks
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_lasso_results['k_fold_age'], net7_lasso_results['predictions'], 1)
ax1.plot(net7_lasso_results['k_fold_age'], n7_a + n7_b * net7_lasso_data['age'], color='red')

ax2.scatter(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_lasso_results['k_fold_age'], net17_lasso_results['predictions'], 1)
ax2.plot(net17_lasso_results['k_fold_age'], n17_a + n17_b * net17_lasso_data['age'], color='red')

ax1.set_title('7 networks')
ax2.set_title('17 networks')
plt.show()

# %%
#and same again for the correlation connectome networks:
f, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], alpha=0.5)
n7_b, n7_a = np.polyfit(net7_corr_results['k_fold_age'], net7_corr_results['predictions'], 1)
ax1.plot(net7_corr_results['k_fold_age'], n7_a + n7_b * net7_corr_data['age'], color='red')

ax2.scatter(net17_corr_results['k_fold_age'],net17_corr_results['predictions'], alpha=0.5)
n17_b, n17_a = np.polyfit(net17_corr_results['k_fold_age'], net17_corr_results['predictions'], 1)
ax2.plot(net17_corr_results['k_fold_age'], n17_a + n17_b * net17_corr_data['age'], color='red')

ax1.set_title('7 networks')
ax2.set_title('17 networks')
plt.show()

# %%
#looking at sparcity of connectomes Ive got
cuttoff = 0.65 #max to still get 1 edge is 0.69
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
#TO DO: maybe something thats the opposite of sparsity, like looking at generally strong edges? 

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
p = (phenotype['p_factor_mcelroy_harmonized_all_samples'])
p = np.array(p)
print(len(p))
mask = []
for val in p:
    mask.append(not math.isnan(val))
p_clean = p[mask]
print(len(p_clean))
phenotype_clean = phenotype[mask]

# %%
