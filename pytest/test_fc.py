import pytest
import subprocess
import sys
import os
import pandas as pd
import nitime
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

from model_fc import models
sys.path.insert(0, os.path.abspath('../scripts/processing'))
from calc_fc import split_kfold, fit_model

def test_arbitrary():
    assert 1 == 1

def test_fc_runs():
    script_path = os.path.abspath(os.path.join(script_dir, "..", "scripts", "processing", 'calc_fc.py'))
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Script crashed with error:\n{result.stderr}"

def test_fc_split():

    data_path = os.path.join(nitime.__path__[0], 'data')
    fname = os.path.join(data_path, 'fmri_timeseries.csv')
    dataframe = pd.read_csv(fname, dtype=float, delimiter=',').T
    data = np.array(dataframe)
    assert not dataframe.empty

    splits_len = 6
    for cv_kind in ['random', 'blocks']:
        splits, timeseries = split_kfold(cv_kind, data, splits_len)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            assert fold_idx <= splits_len
            train_ts = timeseries[train_idx, :]
            test_ts = timeseries[test_idx, :]
            assert not train_ts.size == 0
            assert not test_ts.size == 0

def test_fc_on_nitimedata():
    
    data_path = os.path.join(nitime.__path__[0], 'data')
    fname = os.path.join(data_path, 'fmri_timeseries.csv')
    dataframe = pd.read_csv(fname, dtype=float, delimiter=',').T
    data = np.array(dataframe)
    assert not dataframe.empty
    
    n_rois = len(data[0])
    cv_kind = 'random'
    splits_len = 6
    splits, timeseries = split_kfold(cv_kind, data, splits_len)

    results_list=[]
    for model_str in ['correlation', 'lassoCV', 'uoiLasso']:
        results_dict = fit_model(model_str, splits, timeseries, 1000, n_rois, 10)
        assert results_dict
        results_list.append(results_dict)
    #TO DO: test that they are different

def test_fc_consistency(): 
    """Test whether calc_fc is outputting a consistent matrix. 
    """
    
    #read in exemplar matrix
    
    #compare expemplar matrix to calculated matrixs
