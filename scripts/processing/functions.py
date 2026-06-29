
import glob
import os
import os.path as op

import pandas as pd
import numpy as np
import math
import scipy
import statistics
from copy import deepcopy
import pickle as pkl

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
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

def get_connectomes(con_model, con_filepath, sub_list, network=False):
    connectomes = {}
    for sub in sub_list:
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
                connectome = connectome.to_numpy()                    
                if not network:
                    np.fill_diagonal(connectome, 0)
                    if con_model == 'correlation_random':
                        connectome = np.tril(connectome, k=-1)
                connectome = connectome.ravel()
                connectomes[sub] = connectome

    connectomes = pd.DataFrame(connectomes).transpose()
    
    connectomes.rename(columns={num: f'edge_{num}' for num in range(10000)}, inplace=True)
    connectomes.index = np.int64(connectomes.index)
    connectomes.index.name = 'participant_id'
    return connectomes


def get_networks(con_model, network_labels, con_filepath, sub_list):
    connectomes = {}
    for sub in sub_list:
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
                connectome['network'] = network_labels
                connectome = connectome.groupby('network', as_index=False).mean()
                connectome.drop('network', axis=1, inplace=True)
                connectome = connectome.T
                connectome['network'] = network_labels
                connectome = connectome.groupby('network', as_index=False).mean()
                connectome.drop('network', axis=1, inplace=True)
                connectome = connectome.T
                connectome = connectome.to_numpy()
                connectome = connectome.ravel()
                connectomes[sub] = connectome
    connectomes = pd.DataFrame(connectomes).transpose()
    num_networks = len(set(network_labels))
    net_names = []
    seen = set()
    for item in network_labels:
        if item not in seen:
            net_names.append(item)
            seen.add(item)
    names = [f"{x}_to_{y}" for x in net_names for y in net_names]
    connectomes.rename(columns={num: names[num] for num in range(num_networks**2)}, inplace=True)
    connectomes.index = np.int64(connectomes.index)
    connectomes.index.name = 'participant_id'
    return connectomes


def get_timeseries_pred(con_model, con_filepath, sub_list):
    timeseries_pred = {}
    for sub in sub_list:
        for file in glob.glob(op.join(con_filepath,
                                 con_model,
                                 f'sub-{sub}',
                                 '*results.pkl')):
            with open(file, 'rb') as i:
                loaded_data = pkl.load(i)
                fold_data = {}
                for fold in range(len(loaded_data.keys())):
                    fold_data[fold] = np.mean([loaded_data[f'fold_{fold}'][f'node_{node}']['test_r2'] for node in range(100)])
                timeseries_pred[sub] = statistics.mean(fold_data.values())
    return timeseries_pred


def cpm(X, y, folds, correlate, alpha, cpm_type, model_obj, corr_matricies):
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
        assert cpm_type in ('total', 'positive', 'negative'), "Use 'positive', 'negative', or 'total' to select edges used"
        pl = Pipeline([('scaler', StandardScaler()), ('model', model_obj)])
        pl.fit(train_sub_sums[[cpm_type]], train_y)
        results[f'fold {fold} stats'] = {'test': pl.score(test_sub_sums[[cpm_type]], test_y), 'train': pl.score(train_sub_sums[[cpm_type]], train_y)}
        predictions.extend(pl.predict(test_sub_sums[[cpm_type]]))
        k_fold_y.extend(test_y)
        fold += 1
    
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['MAE'] = mean_absolute_error(k_fold_y, predictions)
    results['predictions'] = predictions
    results['r'] = np.corrcoef(k_fold_y, predictions)[0, 1]
    results['residuals'] = [y - yprime for y, yprime in zip(results['k_fold_age'], results['predictions'])]
    return(results)


def fit_model(X, y, model_obj, folds=5, component=None, verbose=False, weights=False):
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
    coefs = []
    for fold, (train_index, test_index) in enumerate(group_kf.split(X, y, groups)):
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]

        pl.fit(train_X, train_y)      
        results[f'fold {fold} stats'] = [{'test': pl.score(test_X, test_y), 'train': pl.score(train_X, train_y)}]
        if weights:
            coefs.append(list(model_obj.coef_))
        predictions.extend(pl.predict(test_X))
        k_fold_y.extend(test_y)
        if verbose:
            print(f'fold {fold} complete')
    if weights:
        avg_coefs = np.array(coefs)
        results['avg_coefs'] = avg_coefs.mean(axis=0)
    results['k_fold_age'] = k_fold_y
    results['full_r2'] = r2_score(k_fold_y, predictions)
    results['MAE'] = mean_absolute_error(k_fold_y, predictions)
    results['predictions'] = predictions
    results['r'] = np.corrcoef(k_fold_y, predictions)[0, 1]
    results['residuals'] = [y - yprime for y, yprime in zip(results['k_fold_age'], results['predictions'])]
    return(results)


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
        estimator=None,
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
        self.estimator_ = estimator
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

        estimator = self.estimator_
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
        if estimator is not None: #HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERE
            estimator = MultiOutputRegressor(estimator)
            estimator.fit(covariate_fit, X)
            self.weights_ = []
            for model in estimator.estimators_:
                self.weights_.append(model.coef_)
            self.estimator_ = estimator
        else:
            self.weights_ = lstsq(covariate_fit, X)[0]

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

        estimator = self.estimator_
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
        if estimator is not None:
            X_new = X - estimator.predict(covariate_transform)
        else:
            X_new = X - covariate_transform.dot(self.weights_) # CHANGE HERE TO TRUE - PREDICTED #################################################

        # Ensure no NaNs in output
        X_new = np.nan_to_num(X_new)

        return X_new