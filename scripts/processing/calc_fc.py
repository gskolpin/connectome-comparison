# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: FC
#     language: python
#     name: fc
# ---

# This code expects to find parcellated timeseries data in `f'deriv_dir/timeseries/{proc_type}/sub-{sub_id}/sub-{sub_id}*ses-1*timeseries.ptseries.nii'`. 
#
#

# +
import sys
import os
import numpy.matlib
from numpy.matlib import repmat
import nibabel as nib
import json
from sklearn.exceptions import ConvergenceWarning
import sklearn
import warnings
import time
import nilearn
from pyuoi.utils import log_likelihood_glm, AIC, BIC
import numpy as np
import argparse
import os
import os.path as op
from glob import glob
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lars_path
import pickle

from model_fc.models import init_model, run_model


# -


def get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id): 
    if 'hcp' in fc_data_path: 
        if run_id == None: 
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}*ses-{ses_id}*ptseries.nii'))
            
        elif run_id == '1': 
            #YES, future self, it is supposed to be run-{ses_id}, 
            #not run-{run_id}. 
            #RL scan was always ran first.
            #This is consistent with XCPDs HCP renaming code.
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}_dir-RL_*run-{ses_id}**ptseries.nii'))
            
        elif run_id == '2': 
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}_dir-LR_*run-{ses_id}**ptseries.nii'))
    else: 
        ts_file = glob(op.join(fc_data_path, 
                'derivatives', 
                'timeseries',
                proc_type,
                f'sub-{sub_id}',
                f'sub-{sub_id}*task-{task_id}_*ptseries.nii')) #added the singleband here so it should run for participants with multiple scans

        return ts_file


def filter_ts_files(ts_files):
    #if multiple rest scans, selecting out the singeband one, if none are singleband, just select the first one
    if len(ts_files) > 1:
        singlebands = [file for file in ts_files if 'singleband' in file]
        if singlebands:
            ts_files = [singlebands[0]]
        else:
            ts_files = [ts_files[0]]

    assert len(ts_files) == 1 
    return ts_files


def read_cifti(file): 
    time_series = nib.load(file).get_fdata(dtype=np.float32)
    if n_rois != time_series.shape[1]: 
        time_series = time_series[:, :n_rois]
    return time_series


def set_warnings_filters():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def split_kfold(cv, time_series, n_folds, seed=None):
    if seed is not None: 
        pass
    if cv == 'random': 
        
        kfolds = KFold(n_splits=n_folds,
                      shuffle=False)
        
        splits = kfolds.split(X=time_series)
        
    if cv == 'blocks':
        group =  repmat(np.arange(1, n_folds+1), 
                        int(time_series.shape[0]/n_folds), 1).T.ravel()
        
        group = np.concatenate([group, np.ones(time_series.shape[0] - group.shape[0]) * np.max(group)]) 

        
        kfold = GroupKFold(n_splits=n_folds)
        
        splits = kfold.split(X=time_series, groups=group) 
        
    if cv == 'timeseries': 
        tscv = TimeSeriesSplit()
        splits = tscv(tscv.split(X=time_series))
    
    if cv == 'task': 
        task_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        'xcpd',
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{cv_task}_*space*ptseries.nii'))[0]
        task_scan =  nib.load(task_file).get_fdata(dtype=np.float32)[:, :n_rois]
        time_series = np.append(time_series, task_scan, axis=0)

        ses_len = int(len(time_series)/2)
        ses_idx = np.concat([repmat(0, 1, ses_len),
                             repmat(1, 1, ses_len)],axis=-1).ravel()
        ses = sklearn.model_selection.PredefinedSplit(ses_idx)
        splits = ses.split(time_series)
        
    return splits, time_series


def fit_model(model_str, splits, time_series, max_iter, n_rois, random_state, sub_id='Test', ses_id='Test'):
    results_dict = {}    
    print(f"Calculating {model_str} FC for {sub_id} {ses_id}")
    
    scaler = StandardScaler()
    model = init_model(model_str, max_iter, random_state)
    print(model)
    
    if model_str in ["lassoCV", "uoiLasso", "enet", "lassoBIC", "ridgeCV"] :
        print(model_str) 
    
        for fold_idx, (train_idx, test_idx) in enumerate(splits): 
            
            print(f"Fold {fold_idx}:")
            print(f"  Train: range={train_idx[0]}, {train_idx[-1]}")
            print(f"  Test:  index={test_idx[0]}, {test_idx[-1]}")
    
            train_ts = time_series[train_idx, :]
            test_ts = time_series[test_idx, :]
            
            scaler = StandardScaler()
            train_ts = scaler.fit_transform(train_ts)
            test_ts = scaler.transform(test_ts)
    
            start_time = time.time()
            results_dict[f'fold_{fold_idx}'] = run_model(train_ts, test_ts,
                                                         n_rois, model=model)
            results_dict[f'fold_{fold_idx}']['model'] = model
            print(time.time() - start_time, ' seconds')    
            print(model)
                
            return results_dict 
        
    elif model_str in ['correlation', 'tangent', 'partial_correlation']: 
        corr_mat = model.fit_transform(np.array(time_series))[0]
        np.fill_diagonal(corr_mat, 1)
    
        results_dict['fc_matrix'] = corr_mat
        results_dict['model'] = model

    return results_dict


if __name__ == "__main__":
    args = argparse.Namespace()

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    #https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter
    
    parser.add_argument('--sub_id',default='sub-3891810238')
    parser.add_argument('--ses_id', default=1)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--task_id', default='rest') 
    
    parser.add_argument('--atlas_spec', default='fsLR_seg-4S156Parcels_den-91k')
    parser.add_argument('--n_rois', default=100, type=int) #default for hcp; 
    parser.add_argument('--n_trs', default=1200, type=int) #default for hcp;
    parser.add_argument('--n_folds', default=7)
    #must be larger than 6 for PNC
    parser.add_argument('--model', default='lassoCV') 
    parser.add_argument('--cv', default='task')
    parser.add_argument('--proc_type', default='xcpd') 
    
    #TO DO: DEFAULT SHOULD BE BASED ON GSCRATCH OR PSCRATCH
    parser.add_argument('--fc_data_path', 
                        default='/gscratch/psych/gkolpin/data/pnc_xcpd_4S156Parcels')
    parser.add_argument('--max_iter', default=1000) 
    parser.add_argument('--cv_task', default='idemo')

    #TO DO: config file to get this
    sys.path.append(os.path.dirname('/mmfs1/gscratch/escience/gkolpin/model-fc/src/model_fc'))

    set_warnings_filters()
    
    #hack argparse to be jupyter friendly AND cmdline compatible
    try: 
        os.environ['_']
        args = parser.parse_args()
        notebook = False
    except KeyError: 
        args = parser.parse_args([])
        notebook = True
        #TO DO: here too
        results_path = '/gscratch/scrubbed/gkolpin/xcpd_output/pnc_xcpd_4S156Parcels/derivatives/connectivity-matrices/xcpd'
    
    args.sub_id = str(args.sub_id).replace('sub-', '')
    sub_id = args.sub_id
    args.ses_id = str(args.ses_id).replace('ses-', '')
    ses_id = args.ses_id
    task_id = args.task_id
    atlas_spec = args.atlas_spec 
    n_rois = args.n_rois
    n_trs = args.n_trs
    n_folds = args.n_folds
    model_str = args.model
    cv = args.cv
    fc_data_path = args.fc_data_path
    proc_type = args.proc_type
    cv_task = args.cv_task
    
    max_iter = args.max_iter
    run_id = args.run_id
    random_state = 1
    print(args)

    args_dict = vars(args) 

    #TO DO: HCP file names? will they work with this
    ts_files = get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id)
    ts_files = filter_ts_files(ts_files)
    

    if notebook == False:
        #TO DO: CHANGED TO GET A TEST RESULT PATH, CHANGE BACK LATER
        results_path = op.join(fc_data_path, 
                        'derivatives',
                        'connectivity-matrices',
                        proc_type,
                        f"{model_str.replace('-', '')}_{cv}",
                        f'sub-{sub_id}')
    
    os.makedirs(results_path, exist_ok=True)
    
    
    print(f"Found {len(ts_files)} rest scans for subject {sub_id}.") 
    
    print(f"Saving results to {results_path}.")

    time_series = read_cifti(ts_files[0])
    
    splits, time_series = split_kfold(cv, time_series, n_folds)

    results_dict = fit_model(model_str, splits, time_series, max_iter, n_rois, random_state, sub_id, ses_id)
    
    file = f"sub-{sub_id}_task-{task_id}_ses-{ses_id}_{atlas_spec}_model-{model_str.replace('-', '')}_results.pkl"
        
    with open(op.join(results_path,file), 'wb') as f:
        pickle.dump(results_dict, f) 
    
    config_file = file.replace('_results.pkl', '_results.json')
    with open(op.join(results_path, config_file), "w") as f:
        json.dump(args_dict, f, indent=4)
