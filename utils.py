import numpy as np
import pandas as pd
import tqdm
from scipy.stats import norm
from scipy.special import expit
import scipy

def draw_from_signal_distribution(n, phi, delta, sigma_g=1):
    # Draws samples from the x distribution (so this is in signal space, not probability space). 
    n_positive = np.random.binomial(n=n, p=phi)
    n_negative = n - n_positive
    signal_samples = (list(np.random.normal(size=n_positive, loc=delta, scale=sigma_g)) + 
                    list(np.random.normal(size=n_negative, loc=0, scale=1)))
    return pd.DataFrame({"x":signal_samples, 
                         "y":[0] * n_negative + [1] * n_positive})


def signal_to_p(x, phi, delta, sigma_g=1):
    # Converts x -> p. Translated right out of the original R code. 
    p = phi * scipy.stats.norm.pdf(x, loc=delta, scale=sigma_g) / (phi * scipy.stats.norm.pdf(x, loc=delta, scale=sigma_g) + (1 - phi) * scipy.stats.norm.pdf(x, loc=0, scale=1))
    alternate_p = 1/(1 + np.exp(-delta * x + delta ** 2 / 2) * (1 - phi)/phi) # expression used in paper - should be the same. 
    assert np.allclose(p, alternate_p)
    return p

def draw_from_discriminant_distribution(n, phi, delta, sigma_g=1):
    x = draw_from_signal_distribution(n, phi, delta, sigma_g)
    return signal_to_p(x["x"], phi, delta, sigma_g)


from collections import Counter
def generate_chain_data(M, N, K, last_feature_idx_to_use, verbose=False, thresholds=None, deltas=None, beta_X=None, reveal_penultimate=False, seed=123):
    # print("reveal_penultimate", reveal_penultimate)
    # M rows, N columns, K thresholds in the chain. 
    assert K >= 1
    assert len(last_feature_idx_to_use) == K
    assert last_feature_idx_to_use[-1] == N
    if thresholds is None:
        thresholds = sorted(list(np.random.uniform(size=K)))
    if deltas is None:
        deltas = sorted(list(np.random.uniform(size=K) * 3))
    # print('thresholds are', thresholds)
    # print('deltas are', deltas)

    np.random.seed(seed)
    X = np.random.normal(size=(M, N))
    X[:, 0] = 1 # set the first column to 1.
    if beta_X is None:
        beta_X = np.random.normal(size=(N,))
    latent_params = {'thresholds':thresholds, 'deltas':deltas, 'beta_X':beta_X}
    observed_data = {'stage':[], 'last_feature_idx_to_use':last_feature_idx_to_use, 'y':[], 'X':X, 'M':M, 'N':N, 'K':K}
    phis = []
    for i in tqdm.tqdm(range(M)):
        current_stage = 1
        for k in range(K):
            threshold = thresholds[k]
            delta = deltas[k]
            # feature_indices_for_stage = range(last_feature_idx_to_use[k])
            phi = expit(X[i, :last_feature_idx_to_use[k]].dot(beta_X[:last_feature_idx_to_use[k]]))
            if k == 0: # only store the first phi
                phis.append(phi)
            discriminant_draw = draw_from_discriminant_distribution(n=1, phi=phi, delta=delta)[0]
            # print(discriminant_draw)
            if discriminant_draw >= threshold:
                if verbose:
                    print('person %i, k = %i, discriminant draw %2.3f > threshold %2.3f PASSED' % (i, k, discriminant_draw, threshold))
                current_stage += 1
            else:
                if verbose:
                    print('person %i, k = %i, discriminant draw %2.3f < threshold %2.3f FAILED' % (i, k, discriminant_draw, threshold))
                break
        observed_data['stage'].append(current_stage)
        y = -1
        if current_stage == K + 1:
            y = 1 if np.random.random() < discriminant_draw else 0
        if current_stage == K and reveal_penultimate:
            y = 1 if np.random.random() < discriminant_draw else 0
        observed_data['y'].append(y)
        if verbose:
            print("final stage", current_stage, 'discriminant draw', discriminant_draw, 'y', y)
    # print("Number making it to each stage",Counter(observed_data['stage']))
    # print("final y distribution", Counter(observed_data['y']))
    # print(observed_data)
    return latent_params, observed_data, phis



def generate_chain_data_with_express_lane(
    M, N, K, last_feature_idx_to_use,
    verbose=False, thresholds=None, deltas=None, beta_X=None,
    reveal_penultimate=False,
    express_lane=False,
    seed=123
):
    # print("reveal_penultimate", reveal_penultimate)
    # print("express_lane      ", express_lane)

    assert K >= 1
    assert len(last_feature_idx_to_use) == K
    assert last_feature_idx_to_use[-1] == N

    if thresholds is None:
        thresholds = sorted(list(np.random.uniform(size=K)))
    if deltas is None:
        deltas = sorted(list(np.random.uniform(size=K) * 3))
    if beta_X is None:
        beta_X = np.random.normal(size=(N,))

    # print("thresholds are", thresholds)
    # print("deltas are", deltas)
    # print("last_feature_idx_to_use are", last_feature_idx_to_use)

    np.random.seed(seed)
    X = np.random.normal(size=(M, N))
    X[:, 0] = 1  # intercept column

    observed_data = {
        'stage': [],
        'took_express': [],         
        'last_feature_idx_to_use': last_feature_idx_to_use,
        'y': [],
        'hidden_y': [],
        'X': X,
        'M': M,
        'N': N,
        'K': K
    }
    latent_params = {'thresholds': thresholds, 'deltas': deltas, 'beta_X': beta_X}
    phis = []

    for i in tqdm.tqdm(range(M)):
        current_stage = 1
        took_express_flag = 0
        discriminant_draw = None  

        for k in range(K):
            threshold = thresholds[k]
            delta = deltas[k]

            phi = expit(X[i, :last_feature_idx_to_use[k]].dot(
                        beta_X[:last_feature_idx_to_use[k]]))
            if k == 0:
                phis.append(phi)

            discriminant_draw = draw_from_discriminant_distribution(
                                    n=1, phi=phi, delta=delta)[0]
            

            if (k == 0 and express_lane and K >= 2
                    and discriminant_draw >= thresholds[1]):
                took_express_flag = 1
                current_stage = K + 1          
                if verbose:
                    print(f'person {i}: express lane, draw {discriminant_draw:.3f} '
                          f'>= threshold2 {thresholds[1]:.3f}')
                break  # skip the remaining stages

            if discriminant_draw >= threshold:
                if verbose:
                    print(f'person {i}, k={k}, draw {discriminant_draw:.3f} '
                          f'> threshold {threshold:.3f}  PASSED')
                current_stage += 1
            else:
                if verbose:
                    print(f'person {i}, k={k}, draw {discriminant_draw:.3f} '
                          f'< threshold {threshold:.3f}  FAILED')
                break

        y = -1
        if current_stage == K + 1:
            y = 1 if np.random.random() < discriminant_draw else 0
            hidden_y = y
        else:
            hidden_y = 1 if np.random.random() < phis[-1] else 0
            

        observed_data['stage'].append(current_stage)
        observed_data['took_express'].append(took_express_flag)
        observed_data['y'].append(y)
        observed_data['hidden_y'].append(hidden_y)

        if verbose:
            print("final stage", current_stage,
                  "discriminant draw", discriminant_draw,
                  "y", y, "express", took_express_flag)

    print("Number making it to each stage", Counter(observed_data['stage']))
    print("final y distribution          ", Counter(observed_data['y']))
    print("took express count            ", Counter(observed_data['took_express']))
    print("hidden y distribution          ", Counter(observed_data['hidden_y']))

    return latent_params, observed_data, phis




def generate_thresholds_k(K):
    """
    Generates K thresholds in probability space following the specified generative process.
    
    Args:
        K (int): Number of thresholds to generate.
        
    Returns:
        thresholds_in_p_space_sorted (np.ndarray): Sorted array of K thresholds between 0 and 1.
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    
    # Step 1: Sample logit_threshold_0 from Normal(-1, 1)
    logit_threshold_0 = np.random.normal(loc=-1, scale=1)
    
    # Step 2: Sample K-1 gaps from Half-Normal with scale sqrt(0.5)
    gaps = np.abs(np.random.normal(loc=0, scale=np.sqrt(0.5), size=K-1))
    
    # Step 3: Compute cumulative logit thresholds
    logit_thresholds_in_p_space = np.concatenate(([logit_threshold_0], logit_threshold_0 + np.cumsum(gaps)))
    
    # Step 4: Apply sigmoid to transform to probability space
    thresholds_in_p_space = expit(logit_thresholds_in_p_space)
    
    # Sort the thresholds in ascending order
    thresholds_in_p_space_sorted = np.sort(thresholds_in_p_space)
    
    return thresholds_in_p_space_sorted


def generate_deltas_k(K):
    """
    Generates K deltas following the specified generative process.
    
    Args:
        K (int): Number of deltas to generate. Must be at least 1.
        
    Returns:
        deltas_sorted (np.ndarray): Sorted array of K deltas.
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    
    # Step 1: Sample deltas_0 from Half-Normal with scale sqrt(0.1)
    deltas_0 = np.abs(np.random.normal(loc=0, scale=np.sqrt(0.1)))
    
    if K == 1:
        deltas = np.array([deltas_0])
    else:
        # Step 2: Sample K-1 gaps from Half-Normal with scale sqrt(0.3)
        gaps = np.abs(np.random.normal(loc=0, scale=np.sqrt(0.3), size=K-1))
        
        # Step 3: Compute cumulative deltas
        deltas = np.concatenate(([deltas_0], deltas_0 + np.cumsum(gaps)))
    
    deltas_sorted = np.sort(deltas)
    
    return deltas_sorted
