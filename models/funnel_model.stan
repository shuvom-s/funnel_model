data {
int<lower=1> M; // number of observations
int<lower=1> N; // number of features
int<lower=1> K; // number of stages
array[K] int<lower=1> last_feature_idx_to_use;
array[M] int<lower=-1, upper=1> y; // outcome
matrix[M, N] X; // features
array[M] int<lower=1, upper=K + 1> stage; // stage
}

parameters {	
vector[N] beta_X; // coefficients for features
//positive_ordered[K] thresholds_in_p_space;
//positive_ordered[K] deltas; // deltas
real<lower=0, upper=0.5> threshold_base;
array[K-1] real<lower=0, upper=0.5> threshold_gap;

real<lower=0, upper=5> delta_base;
array[K-1] real<lower=0, upper=5> delta_gap;
}

transformed parameters {
  // Build group-specific ordered threshold vectors.
  vector[K] thresholds_in_p_space;
  thresholds_in_p_space[1] = threshold_base;
  thresholds_in_p_space[2] = threshold_base + threshold_gap[1];

  vector[K] deltas;
  deltas[1] = delta_base;
  for (k in 2:K)
    deltas[k] = deltas[k-1] + delta_gap[k-1];
}

model { 
    beta_X ~ normal(1, 1);
    // thresholds_in_p_space ~ normal(0.4, 0.1);
    // deltas ~ normal(0, 1);
    // Priors for group-specific thresholds.
    threshold_base ~ normal(0, 0.5);
    for (k in 1:(K - 1))
        threshold_gap[k] ~ normal(0, 0.5);


    // Priors for deltas using our offset formulation.
    delta_base ~ normal(0, 0.5);
    for (k in 1:(K - 1))
        delta_gap[k] ~ normal(0, 0.5);
    
    for(i in 1:M){
        for(k in 1:min(stage[i], K)){
            real eps = 1e-10;
            real partial_xbeta = X[i, 1:last_feature_idx_to_use[k]] * beta_X[1:last_feature_idx_to_use[k]];
            real phi = inv_logit(partial_xbeta);
            real threshold_for_obs = fmin(1 - eps, fmax(thresholds_in_p_space[k], eps));

            real log_odds_ratio = partial_xbeta + log((1 - threshold_for_obs + eps) / (threshold_for_obs + eps));
            real threshold_in_signal_space = (square(deltas[k]) - 2 * log_odds_ratio) / (2 * deltas[k]);

            real cdf_pos = Phi(threshold_in_signal_space - deltas[k]);
            real cdf_neg = Phi(threshold_in_signal_space);
            real mass_above_threshold_from_pos_distribution = phi * (1 - cdf_pos);
            real mass_above_threshold_from_neg_distribution = (1 - phi) * (1 - cdf_neg);


            real search_rate = (mass_above_threshold_from_pos_distribution + mass_above_threshold_from_neg_distribution);
            if(k == stage[i]){
                target += log1m(search_rate);
            } else {
                target += log(search_rate);
            }
            if(k == K && stage[i] == K + 1){
                real hit_rate = mass_above_threshold_from_pos_distribution / search_rate; 
                if(y[i] == 1){target += log(hit_rate);} 
                else {target += log1m(hit_rate);}
                }
        }
    } 
} 

generated quantities {
  array[K] vector[M] risk_pred;        // φ_k[i] = P(Y=1 | X_i)
  array[K] vector[M] search_rate_pred; // P(pass stage k)
  vector[M]          hit_rate_pred;    // P(Y=1 | pass final stage)

  {
    real eps = 1e-10; // numerical stability

    for (k in 1:K) {
      for (i in 1:M) {

        // latent risk
        real eta = X[i, 1:last_feature_idx_to_use[k]]
                 * beta_X[1:last_feature_idx_to_use[k]];
        real phi = inv_logit(eta);
        risk_pred[k][i] = phi;

        real thr_p   = fmin(1 - eps, fmax(thresholds_in_p_space[k], eps));
        real log_odds_ratio = eta
                            + log( (1 - thr_p + eps) / (thr_p + eps) );
        real thr_sig = (square(deltas[k]) - 2 * log_odds_ratio)
                       / (2 * deltas[k]);

        real cdf_pos = Phi(thr_sig - deltas[k]);
        real cdf_neg = Phi(thr_sig);
        real m_pos   = phi * (1 - cdf_pos);
        real m_neg   = (1 - phi) * (1 - cdf_neg);
        real srate   = m_pos + m_neg + 1e-12;   // underflow guard
        search_rate_pred[k][i] = srate;

        if (k == K)
          hit_rate_pred[i] = m_pos / srate;
      }
    }
  }

  int   n_rep_pass2       = 0;
  int   n_rep_pass3       = 0;
  int   n_rep_pass3_hit   = 0;
  int   n_rep_pass3_total = 0;

  for (i in 1:M) {
    int stage_rep = 1; 
    int y_rep     = -1;

    for (k in 1:K) {
      real srate = search_rate_pred[k][i];
      real hr    = (k == K) ? hit_rate_pred[i] : 0;

      srate = fmin(1 - 1e-12, fmax(1e-12, srate));
      hr    = fmin(1 - 1e-12, fmax(1e-12, hr));

      if (bernoulli_rng(srate)) {
        if (k == K) {
          stage_rep = K + 1;
          y_rep     = bernoulli_rng(hr) ? 1 : 0;
        }
      } else {
        stage_rep = k;
        break;
      }

    }

    if (stage_rep >= 2) n_rep_pass2 += 1;
    if (stage_rep >= 3) {
      n_rep_pass3       += 1;
      n_rep_pass3_total += 1;
      if (y_rep == 1) n_rep_pass3_hit += 1;
    }
  }


  real prop_pass2_rep = n_rep_pass2 / (1.0 * M);
  real prop_pass3_rep = n_rep_pass3 / (1.0 * M);
  real hit_rate_rep   = n_rep_pass3_total > 0
                        ? n_rep_pass3_hit / (1.0 * n_rep_pass3_total)
                        : 0;

  real prop12_search_rate = n_rep_pass2 / (1.0 * M);
  real prop23_search_rate = n_rep_pass2 > 0
                            ? n_rep_pass3 / (1.0 * n_rep_pass2)
                            : 0;
}
