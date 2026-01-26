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
from copy import deepcopy
import pickle as pkl
import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

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

# %%
sublist = []
with open(sublist_filepath, 'r') as file:
    sublist = file.read().splitlines()
print(len(sublist))


# %%
def get_connectomes(con_model=con_model, con_filepath=con_filepath, sublist=sublist):
    connectomes = {}
    for sub in sublist:
        for file in glob.glob(op.join(con_filepath,
                                 con_model,
                                 f'sub-{sub}',
                                 '*results.pkl')):
            with open(file, 'rb') as i:
                loaded_data = pkl.load(i)
                if con_model == 'lassoBIC_blocks':
                    #currently getting just the fold 0 model, I know median of them was mentioned?
                    #remove diagonal here, and down
                    fold_0_model = loaded_data['fold_0']['fc_matrix']
                    connectomes[sub] = fold_0_model.ravel()
                if con_model == 'correlation_random':
                   # If I want to tril them, which intuitively I feel like I should, but maybe it doesn't change much
                   # temp = np.tril(temp, k=-1)
                   #temp = temp.ravel() 
                   connectomes[sub] = loaded_data['fc_matrix'].ravel()
    #connectomes = pd.DataFrame(connectomes).transpose()
    #print(np.shape(connectomes))
    #pprint.pprint(connectomes)
    connectomes = pd.DataFrame(connectomes).transpose()
    connectomes.rename(columns={num: f'edge_{num}' for num in range(10000)}, inplace=True)
    connectomes.index = np.int64(connectomes.index)
    connectomes.index.name = 'participant_id'
    return(connectomes)


# %%
lasso_connectomes = get_connectomes()
lasso_connectomes.head()
print(np.shape(lasso_connectomes))

# %%
corr_connectomes = get_connectomes(con_model='correlation_random')
corr_connectomes.head()

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
#I will just exclude them for now, but this should be solved once we get the task scans going

# %%
phenotype = pd.read_csv(p_filepath, delimiter = '\t', header=0)

# %%
phenotype.set_index('participant_id', inplace=True, drop=False)
phenotype.rename(columns={'participant_id': 'id'}, inplace=True)
None

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
pprint.pprint(rand100_phenotype)


# %%
def eval_metrics(X_train, y_train, X_test, y_test, model):
    """Calculates R2 scores for FC models."""

    test_rsq = r2_score(y_test, model.predict(X_test))
    train_rsq = r2_score(y_train, model.predict(X_train))

    return (test_rsq, train_rsq)


# %%
def fit_lasso(X, y, folds, model_str):
    kf = KFold(folds)
    fold = 0
    results = {}
    predictions = []
    k_fold_y = []
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in kf.split(X):
        fold += 1
        #print(f" Train: shape={np.shape(train_index)}")
        #print(f" Test: shape={np.shape(test_index)}")
        train_X = X[train_index]
        train_X = scaler.fit_transform(train_X)
        train_y = y[train_index]
        test_X = X[test_index]
        test_X = scaler.transform(test_X)
        test_y = y[test_index]

        if model_str == 'Lasso':
            model = Lasso()
        if model_str == 'LassoCV':
            model = LassoCV() #default is 5
            
        model.fit(train_X, train_y)
        test_rsq, train_rsq = eval_metrics(train_X, train_y, test_X, test_y, model)
        results[f'fold {fold} stats'] = {'eval_metrics': [test_rsq, train_rsq], 'model': deepcopy(model)}
        predictions.extend(model.predict(test_X))
        k_fold_y.extend(test_y)
        print(f'fold {fold} complete')
    #Now I have the predictions for each left out part, really should concatinate
    #those somewhere else and then just report the p.
    results['full_prediction_r2'] = r2_score(k_fold_y, predictions)
    results['predictions'] = predictions
    results['k_fold_age'] = k_fold_y
    return(results)


# %%
#now I need to 1: make sure they are ordered correctly 
# (ideally just merge the dataframes on index... BUT with 2 missing them, need to
# exlude them ig, merge right?
lasso_data = pd.concat([lasso_connectomes, rand100_phenotype], join='inner', axis=1)
print(np.shape(lasso_data))
corr_data = pd.concat([corr_connectomes, rand100_phenotype], join='inner', axis=1)
print(np.shape(corr_data))

# %%
#its okay now
#print(type(rand100_phenotype.index[1]))
#print(type(connectomes.index[1]))
#for index in connectomes.index:
#    assert index in rand100_phenotype.index

# %%
lasso_results = fit_lasso(lasso_data[[f'edge_{num}' for num in range(10000)]], lasso_data['age'], 5, 'LassoCV')
corr_results = fit_lasso(corr_data[[f'edge_{num}' for num in range(10000)]], corr_data['age'], 5, 'LassoCV')

# %%
print(lasso_results['full_prediction_r2'])
print(corr_results['full_prediction_r2'])

# %%
print(lasso_results['fold 1 stats']['eval_metrics'])
print(corr_results['fold 1 stats']['eval_metrics'])

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
def plotage(data):
    histage = data.hist(['age'], bins=np.arange(32) - 0.5)
    plt.xticks(range(8, 24))
    plt.xlim(7, 24)
    plt.grid(False)
    plt.xlabel('age')
    plt.ylabel('count')
    plt.title('PNC age distribution')
    plt.show()
    print('min:',  data['age'].min(), 'max:', data['age'].max())


# %%
plotage(rand100_phenotype)

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
