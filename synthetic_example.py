
import nest_asyncio
nest_asyncio.apply()
import multiprocessing

multiprocessing.set_start_method('fork')
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import norm
from tqdm import tqdm
import os
import pickle
import xarray as xr
import time
from cmdstanpy import CmdStanModel
import cmdstanpy
cmdstanpy.set_cmdstan_path('/data/cb/shuvom/funnels/cmdstan')


import json
from scipy.stats import halfnorm
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.calibration import calibration_curve


import argparse
from utils import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, help='Number of observations', required=False, default=5000)
    parser.add_argument('--N', type=int, help='Number of features', required=False, default=7)
    parser.add_argument('--K', type=int, help='Number of stages in the chain', required=False, default=2)
    parser.add_argument('--samples', type=int, help='Number of samples', required=False, default=500)
    parser.add_argument('--warmup', type=int, help='Number of warmup samples', required=False, default=500)
    parser.add_argument('--seed', type=int, help='Random seed', default=0, required=False)
    parser.add_argument('--thresholds', type=float, nargs='+', help='Thresholds for each stage', default = [0.3, 0.5], required=False)
    parser.add_argument('--deltas', type=float, nargs='+', help='Deltas for each stage', default = [0.1, 0.8], required=False)
    parser.add_argument('--reveal_penultimate', type=bool, help='Reveal penultimate stage', default = False, required=False)
    parser.add_argument('--gpu', type=int, help='GPU to use', default=0, required=False)
    parser.add_argument('--express_lane_and_penultimate', type=bool, help='Use express lane and penultimate stage', default = False, required=False)
    parser.add_argument('--generate_data', type=bool, help='Generate data', default = False, required=False)
    args = parser.parse_args()
    M = args.M
    N = args.N
    K = args.K
    thresholds = args.thresholds
    deltas = args.deltas
    assert len(thresholds) == K
    assert len(deltas) == K
    last_feature_idx_to_use = [4, 7]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    model = CmdStanModel(stan_file="/data/cb/shuvom/funnels/repo/models/funnel_model.stan", force_compile=True)
    

    if args.generate_data:
        seed = 0
        beta_X = np.random.normal(size=(N,))
        print("beta_X", beta_X)
        thresholds = [0, 0]
        thresholds[0] = np.random.uniform(0, 0.5)
        thresholds[1] = thresholds[0] + np.random.uniform(0, 0.5)
        
        deltas = [0, 0]
        deltas[0] = halfnorm.rvs(scale=0.5)
        deltas[1] = deltas[0] + halfnorm.rvs(scale=0.5)

        if not args.express_lane_and_penultimate:
            latent_params, observed_data, phis = generate_chain_data(M=M, N=N, K=K, last_feature_idx_to_use=last_feature_idx_to_use,
                                                            thresholds=thresholds, deltas=deltas, beta_X=beta_X, reveal_penultimate=args.reveal_penultimate, seed=seed)
        else:
            latent_params, observed_data, phis = generate_chain_data_with_express_lane(M=M, N=N, K=K, last_feature_idx_to_use=last_feature_idx_to_use,
                                                            thresholds=thresholds, deltas=deltas, beta_X=beta_X, reveal_penultimate=False, express_lane=False, seed=seed)
    
        y = observed_data['y']  # Outcome variable
        from collections import Counter
        print("Count of values in y:", Counter(observed_data['y']))
        y_np = np.array(y)
        X = observed_data['X']  # Feature matrix
        stage = observed_data['stage']
        stage_np = np.array(stage)

        if np.sum(stage_np==1) < 25 or np.sum(stage_np==2) < 25 or np.sum(stage_np==3) < 25 or np.sum(y_np==0) < 25 or np.sum(y_np==1) < 25 or np.sum(y_np==-1) < 25:
            print("skipping")

    else:
        observed_df = pd.read_csv("/data/cb/shuvom/funnels/repo/example_data.csv")
        observed_data = observed_df.to_dict(orient='list')
        observed_data['last_feature_idx_to_use'] = last_feature_idx_to_use
        observed_data['M'] = M
        observed_data['N'] = N
        observed_data['K'] = K
        observed_data['X'] = observed_df[['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6']].values.tolist()


        
    fit = model.sample(
        data=observed_data, 
        chains=4, 
        parallel_chains=4, 
        iter_warmup=args.warmup, 
        iter_sampling=args.samples, 
        seed=args.seed, 
        show_console=True,
        save_warmup=True,
    )
    summary = fit.summary(percentiles=(2,5,50,95,98))
    print("az summary")
    print(az.summary(fit))
    idata = az.from_cmdstanpy(fit, save_warmup=True)
    mask = (
        summary.index.str.startswith("beta_X") |
        summary.index.str.startswith("thresholds_in_p_space") |
        summary.index.str.startswith("deltas") |
        summary.index.str.startswith("n_rep_pass") |
        summary.index.str.startswith("prop_pass") |
        summary.index.str.startswith("hit_rate_rep") |
        summary.index.str.startswith("prop12_search_rate") |
        summary.index.str.startswith("prop23_search_rate")
    )
    
    print(summary[mask])
    print("thresholds", thresholds)
    print("deltas", deltas)
    print("Count of values in y:", Counter(observed_data['y']))

