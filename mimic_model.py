import sys
import numpy as np
from cmdstanpy import CmdStanModel

from scipy.stats import norm
from sklearn.calibration import CalibrationDisplay

import cmdstanpy
import os
import argparse
import time
import arviz as az
import matplotlib.pyplot as plt
import pickle
import json
# import argparse # Duplicate import removed
import pandas as pd
import datetime
import xarray as xr
from numpy.linalg import qr, inv
from sklearn.calibration import CalibrationDisplay # Added for calibration plot

sys.path.append("/data/cb/shuvom/funnels/chain_modeling")
from utils import *

cmdstanpy.set_cmdstan_path('/data/cb/shuvom/funnels/cmdstan')


def list_groups(mimic_data, group_on):
    return mimic_data[group_on].unique()




def run_inference(sampled_mimic, model_type, warmup_steps=500, sampling_steps=500, data_type="staged"):
    
    K = 2
    features = []
    if data_type == "staged_quadratic":
        # base_features = ['intercept', 'triage_temperature', 'triage_heartrate', 'triage_resprate', 'triage_o2sat', 'triage_sbp', 'triage_dbp', 'triage_pain']
        base_features = ['intercept', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        features = base_features.copy()
        for feature in base_features:
            if feature not in ['intercept']:
                features.append(f"{feature}_sq")
        features.extend(['chiefcom_chest_pain', 'chiefcom_abdominal_pain', 'chiefcom_headache', 'chiefcom_shortness_of_breath', 'chiefcom_back_pain', 'chiefcom_cough', 'chiefcom_nausea_vomiting', 'chiefcom_fever_chills', 'chiefcom_syncope', 'chiefcom_dizziness'])
        features.append('age')
        last_feature_idx_to_use = [25, 26] 
        N = 26

    elif data_type == "staged":
        # features = ['intercept', 'triage_temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'gender_binarized', 'age_normalized']
        features = ['intercept', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        features.extend(['chiefcom_chest_pain', 'chiefcom_abdominal_pain', 'chiefcom_headache', 'chiefcom_shortness_of_breath', 'chiefcom_back_pain', 'chiefcom_cough', 'chiefcom_nausea_vomiting', 'chiefcom_fever_chills', 'chiefcom_syncope', 'chiefcom_dizziness'])
        features.append('age')
        last_feature_idx_to_use = [21, 22] 
        N = 22


    # Binarize the chiefcom features
    chiefcom_features = [f for f in features if f.startswith('chiefcom_')]
    for f in chiefcom_features:
        sampled_mimic[f] = sampled_mimic[f].astype(int)

    # Prepare features and target
    X = sampled_mimic[features]
    y = sampled_mimic['y']
    stage = sampled_mimic['stage']
    stage_np = np.array(stage)
    y_np = np.array(y)
    M = len(sampled_mimic)
    took_express = sampled_mimic['ICU-ized-direct'].values
    # took_express = sampled_mimic['icu_direct'].values

    # Initialize model
    if model_type == "original":
        model = CmdStanModel(stan_file="/data/cb/shuvom/funnels/repo/models/funnel_model_express_search_rate_pred.stan", force_compile=True)
        # X = X.clip(-4, 4) 
        
        # Prepare data for Stan
        observed_data = {
            "M": M,
            "N": N,
            "K": K,
            "last_feature_idx_to_use": last_feature_idx_to_use,
            "X": X.values,
            "y": y.values,
            "stage": stage.values,
            "took_express": took_express.astype(int)
        }

    
    # Run the model
    fit = model.sample(
        data=observed_data, 
        chains=4, 
        parallel_chains=4, 
        iter_warmup=warmup_steps, 
        iter_sampling=sampling_steps, 
        seed=10, 
        show_console=True,
        save_warmup=True,
    )

    search_rate_pred      = fit.stan_variable("search_rate_pred")       
    hit_rate_pred         = fit.stan_variable("hit_rate_pred")          
    hit_rate_stage2_pred  = fit.stan_variable("hit_rate_stage2_pred")    
    hit_rate_skip_pred    = fit.stan_variable("hit_rate_skip_pred")      
    hit_rate_stage3_pred  = fit.stan_variable("hit_rate_stage3_pred")    
    skip_prob_pred        = fit.stan_variable("skip_prob_pred")          

    search_rate_mean   = search_rate_pred.mean(axis=0)   
    hit_rate2_mean     = hit_rate_stage2_pred.mean(axis=0)
    hit_rate_skip_mean = hit_rate_skip_pred.mean(axis=0)
    hit_rate_s3_mean   = hit_rate_stage3_pred.mean(axis=0)
    skip_prob_mean     = skip_prob_pred.mean(axis=0)

    stage_np = np.asarray(stage.values)
    y_np     = np.asarray(y.values)
    texp_np  = np.asarray(took_express).astype(int)

    mask_drop2   = (stage_np == 2) & (y_np != -1)
    hit_12_obs   = (y_np[mask_drop2] == 1).mean() if mask_drop2.any() else np.nan
    hit_2_any_obs = hit_12_obs  # identical in this funnel

    mask_skip    = (texp_np == 1) & (stage_np >= 3) & (y_np != -1)
    hit_13_obs   = (y_np[ mask_skip] == 1).mean() if mask_skip.any() else np.nan

    mask_s3_norm = (stage_np >= 3) & (texp_np == 0) & (y_np != -1)
    hit_123_obs  = (y_np[mask_s3_norm] == 1).mean() if mask_s3_norm.any() else np.nan

    mask_s3_all  = (stage_np >= 3) & (y_np != -1)
    hit_3_any_obs = (y_np[mask_s3_all] == 1).mean() if mask_s3_all.any() else np.nan

    prop_pass2_obs   = (stage_np >= 2).mean()
    prop_pass3_obs   = (stage_np >= 3).mean()
    prop_express_obs = texp_np.mean()

    obs_12 = ((stage_np >= 2) & (texp_np == 0))
    search_rate_12 = obs_12.sum() / len(obs_12)

    mask_s2domain  = (stage_np >= 2) & (texp_np == 0)
    obs_23         = (stage_np[mask_s2domain] >= 3).astype(int)
    search_rate_23 = obs_23.mean() if obs_23.size else np.nan

    ppc_vars = [
        "prop_pass2_rep", "prop_pass3_rep", "prop_express_rep",
        "hit_rate_stage2_rep",  
        "hit_rate_skip_rep",   
        "hit_rate_stage3_rep",  
        "hit_rate_rep",        
        "prop12_search_rate", "prop23_search_rate",
    ]

    idata = az.from_cmdstanpy(
        posterior            = fit,
        posterior_predictive = ppc_vars,
    )


    idata.add_groups({
        "observed_data": xr.Dataset(
            {
                "prop_pass2_rep"      : xr.DataArray(prop_pass2_obs),
                "prop_pass3_rep"      : xr.DataArray(prop_pass3_obs),
                "prop_express_rep"    : xr.DataArray(prop_express_obs),

                "hit_rate_stage2_rep" : xr.DataArray(hit_2_any_obs),   
                "hit_rate_skip_rep"   : xr.DataArray(hit_13_obs),
                "hit_rate_stage3_rep" : xr.DataArray(hit_123_obs),
                "hit_rate_rep"        : xr.DataArray(hit_3_any_obs),

                "prop12_search_rate"  : xr.DataArray(search_rate_12),
                "prop23_search_rate"  : xr.DataArray(search_rate_23),
            }
        )
    })

    print("fitted E[mortality | 1→2]      :", float(fit.stan_variable("hit_rate_stage2_rep").mean()))
    print("fitted E[mortality | 1→3]      :", float(fit.stan_variable("hit_rate_skip_rep").mean()))
    print("fitted E[mortality | 1→2→3]    :", float(fit.stan_variable("hit_rate_stage3_rep").mean()))
    print("fitted E[mortality | stage 3]  :", float(fit.stan_variable("hit_rate_rep").mean()))
    print("fitted E[search rate | 1→2]   :", float(fit.stan_variable("prop12_search_rate").mean()))
    print("fitted E[search rate | 2→3]   :", float(fit.stan_variable("prop23_search_rate").mean()))

    # Generate and display results
    print(fit.summary(percentiles=(2,5,50,95,98)))
    summary = fit.summary(percentiles=(2,5,50,95,98))
    
    idata = az.from_cmdstanpy(fit, save_warmup=True)
    return fit, summary, idata


def run_chain_modeling(model, group_on=None, group_on_value=None, sample_size=None, warmup_steps=500, sampling_steps=500, data_type="staged", path_to_mimic_data=None, path_to_base_results_dir=None):
    
    # mimic = pd.read_csv('/data/cb/scratch/sophia/rsidata/mimic_with_express_hosp.csv')
    mimic = pd.read_csv(path_to_mimic_data)

    # Apply sampling if specified
    if sample_size is not None and sample_size > 0:
        mimic = mimic.sample(sample_size, random_state=1)
    
    mimic['y'] = mimic['hospital_expire_flag'].astype(int)
    

    # Create base results directory based on model type, data type, and feature set
    os.makedirs(path_to_base_results_dir, exist_ok=True)
    os.makedirs(f"{path_to_base_results_dir}/posteriors", exist_ok=True)
    
    # Save run parameters for reference
    run_params = {
        "model": model,
        "group_on": group_on,
        "group_on_value": group_on_value,
        "sample_size": sample_size,
        "warmup_steps": warmup_steps,
        "sampling_steps": sampling_steps,
        "data_type": data_type,
        "data_shape": mimic.shape,
        "features": list(mimic.columns),
        "model_features": ['intercept', 'triage_temperature', 'triage_heartrate', 'triage_resprate', 'triage_o2sat', 'triage_sbp', 'triage_dbp', 'triage_pain', 'acuity'] if data_type == "staged_linear" else
                         ['intercept', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcom_chest_pain', 'chiefcom_abdominal_pain', 'chiefcom_headache', 'chiefcom_shortness_of_breath', 'chiefcom_back_pain', 'chiefcom_cough', 'chiefcom_nausea_vomiting', 'chiefcom_fever_chills', 'chiefcom_syncope', 'chiefcom_dizziness']
    }
    
    with open(f"{path_to_base_results_dir}/run_parameters.json", 'w') as f:
        json.dump(run_params, f, indent=4, default=str)
    
    mimic_groups = {}
    if group_on is None:
        groups = [1]
    else:
        groups = list_groups(mimic, group_on)
        # Remove nan values from groups
        groups = [g for g in groups if pd.notna(g)]
        
    # Filter groups based on the group_on variable
    if group_on == "coarse_race":
        groups = [g for g in groups]
    elif group_on == "gender":
        groups = [g for g in groups]
    
    if group_on_value is not None:
        groups = [group for group in groups if group == group_on_value]
        
    for group in groups:
        print(f"Running inference for group: {group}")
        if group_on is None:
            mimic_groups[group] = mimic
        else:
            mimic_groups[group] = mimic[mimic[group_on] == group]

        sampled_mimic = mimic_groups[group]
        
        fit, summary, idata = run_inference(sampled_mimic, model_type=model, warmup_steps=warmup_steps, sampling_steps=sampling_steps, data_type=data_type)
        print(summary)
        
        posterior_samples = fit.stan_variables()
        np.save(f"{path_to_base_results_dir}/posteriors/beta_X_{group}.npy", posterior_samples['beta_X'])
        np.save(f"{path_to_base_results_dir}/posteriors/thresholds_in_p_space_{group}.npy", posterior_samples['thresholds_in_p_space'])
        np.save(f"{path_to_base_results_dir}/posteriors/deltas_{group}.npy", posterior_samples['deltas'])

    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_on", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="original", choices=["original", "penultimate"], 
                        help="Model type: 'original' or 'penultimate' (default: original)")
    parser.add_argument("--job_name", type=str, default="default")
    parser.add_argument("--group_on_value", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use (default: use all data)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for MCMC (default: 500)")
    parser.add_argument("--sampling_steps", type=int, default=500, help="Number of sampling steps for MCMC (default: 500)")
    parser.add_argument("--data_type", type=str, default="staged_quadratic", choices=["staged_linear", "staged_quadratic"], 
                        help="Type of data to use: 'stage_linear' (standard features) or 'staged_quadratic' (default: staged)")
    parser.add_argument("--path_to_mimic_data", type=str, default=None, help="Path to the mimic data")
    parser.add_argument("--path_to_base_results_dir", type=str, default=None, help="Path to the base results directory")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_chain_modeling(
        model=args.model, 
        group_on=args.group_on, 
        group_on_value=args.group_on_value,
        sample_size=args.sample_size,
        warmup_steps=args.warmup_steps,
        sampling_steps=args.sampling_steps,
        data_type=args.data_type,
        include_gender=args.include_gender,
        path_to_mimic_data=args.path_to_mimic_data,
        path_to_base_results_dir=args.path_to_base_results_dir
    )

# %%


