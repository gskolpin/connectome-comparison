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
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

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
sublist_filepath = '/gscratch/escience/gkolpin/connectome-comparison/data/rand100_sub_list.txt'
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

# %%
lasso_connectomes = get_connectomes(con_model='lassoBIC_blocks')
#lasso_connectomes.head()

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
#for data merging, move things out, be clearer, erorrs can happen there
model = LassoCV(random_state=random_state)
#model = LinearRegression()
lasso_results = fit_model(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], model)
corr_results = fit_model(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], model)
net7_lasso_results = fit_model(net7_lasso_data[net7_lasso_connectomes.columns], net7_lasso_data['age'], model)
net17_lasso_results = fit_model(net17_lasso_data[net17_lasso_connectomes.columns], net17_lasso_data['age'], model)
net7_corr_results = fit_model(net7_corr_data[net7_corr_connectomes.columns], net7_corr_data['age'], model)
net17_corr_results = fit_model(net17_corr_data[net17_corr_connectomes.columns], net17_corr_data['age'], model)

# %%
#Okay soo this can get the coefficient path plots from seemingly where they start changing to all 0
alphas = np.logspace(-7, 0, 50)
coefs = []
for a in alphas:
    model = Lasso(random_state=random_state, alpha=a)
    #model.fit(net7_corr_data[net7_corr_data.columns[:49]], net7_corr_data['age']) #looks fairly normal
    model.fit(net7_lasso_data[net7_corr_data.columns[:49]], net7_lasso_data['age']) #super huge coefficients
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
plt.title("Ridge Coefficients vs Regularization Strength (alpha)")
plt.axis("tight")
plt.show()

# %%
# make some new ones for comparison
print('lasso:', lasso_results['fold 1 stats'], 'full r2:', lasso_results['full_r2']) #looks just like a lot of overfitting
print('correlation:', corr_results['fold 1 stats'], 'full r2:', corr_results['full_r2']) #um huh, maybe just underfitting from not enough subjects??? weird b/c still lots of features
print('lasso 7 networks:', net7_lasso_results['fold 1 stats'], 'full r2:', net7_lasso_results['full_r2'])
print('lasso 17 networks:', net17_lasso_results['fold 1 stats'], 'full r2:', net17_lasso_results['full_r2'])
print('correlation 7 networks:', net7_corr_results['fold 1 stats'], 'full r2:', net7_corr_results['full_r2'])
print('correlation 17 networks:', net17_corr_results['fold 1 stats'], 'full r2:', net17_corr_results['full_r2'])

# %%
data = net7_corr_data
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
age_matrix = np.tril(age_matrix)

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
#TO DO: try and get CPM working (following the shen 2017 paper)

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
#need to update this after making the diagonal 0
sparce_edges = {}
for edge_name, edges in lasso_connectomes.items():
    sparcity = (len(edges) - np.count_nonzero(edges)) / len(edges)
    if sparcity >= 0.9:
        #the term sanity check is kinda great i love it, and need it here but ya seems like im right
        sparce_edges[edge_name] = sparcity
#pprint.pprint(sparce_edges)
print(len(sparce_edges))

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
