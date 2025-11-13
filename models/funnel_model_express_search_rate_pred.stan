
data {
  int<lower=1> M;                              // observations
  int<lower=1> N;                              // features
  int<lower=1> K;                              // stages (must be 2)
  array[K] int<lower=1> last_feature_idx_to_use;
  array[M] int<lower=-1,upper=1> y;            // outcome  (−1 = unobserved)
  matrix[M,N] X;                               // features
  array[M] int<lower=1,upper=K+1> stage;       // 1,2,3
  array[M] int<lower=0,upper=1> took_express;  // 1 = skipped stage-2
}

parameters {
  vector[N] beta_X;

  real<lower=0,upper=0.5>     threshold_base;
  array[K-1] real<lower=0,upper=0.5> threshold_gap;

  real<lower=0,upper=5>       delta_base;
  array[K-1] real<lower=0,upper=5>  delta_gap;
}

transformed parameters {
  vector[K] thresholds_in_p_space;
  vector[K] deltas;

  thresholds_in_p_space[1] = threshold_base;
  for (k in 2:K)
    thresholds_in_p_space[k] = thresholds_in_p_space[k-1] + threshold_gap[k-1];

  deltas[1] = delta_base;
  for (k in 2:K)
    deltas[k] = deltas[k-1] + delta_gap[k-1];
}

model {
  if (K != 2)
    reject("express-lane version requires K = 2");

  //beta_X         ~ normal(0,1);
  beta_X[1] ~ normal(-5.5, 0.25);  // First coefficient has a different prior
  beta_X[2:N] ~ normal(0, 0.25);
  threshold_base ~ normal(0,0.5);
  threshold_gap  ~ normal(0,0.5);
  delta_base     ~ normal(0,0.5);
  delta_gap      ~ normal(0,0.5);

  for (i in 1:M) {
    real eps = 1e-10;

    real eta1 = X[i, 1:last_feature_idx_to_use[1]]
              * beta_X[1:last_feature_idx_to_use[1]];
    real phi1 = inv_logit(eta1);
    phi1      = fmin(1-eps, fmax(eps, phi1));

    // thresholds τ1 < τ2
    real tau1 = fmin(1-eps, fmax(thresholds_in_p_space[1], eps));
    real tau2 = fmin(1-eps, fmax(thresholds_in_p_space[2], eps));

    real log_odds_tau1 = eta1 + log((1-tau1+eps)/(tau1+eps));
    real log_odds_tau2 = eta1 + log((1-tau2+eps)/(tau2+eps));

    real thr1_sig = (square(deltas[1]) - 2*log_odds_tau1) / (2*deltas[1]);
    real thr2_sig = (square(deltas[1]) - 2*log_odds_tau2) / (2*deltas[1]);

    real cdf1_pos = Phi(thr1_sig - deltas[1]);
    real cdf1_neg = Phi(thr1_sig);
    real cdf2_pos = Phi(thr2_sig - deltas[1]);
    real cdf2_neg = Phi(thr2_sig);

    real m_pos_above1 = phi1       * (1 - cdf1_pos);
    real m_neg_above1 = (1 - phi1) * (1 - cdf1_neg);
    real m_pos_above2 = phi1       * (1 - cdf2_pos);
    real m_neg_above2 = (1 - phi1) * (1 - cdf2_neg);

    real skip_mass    = m_pos_above2 + m_neg_above2;              
    real pass12_mass  = (m_pos_above1 + m_neg_above1) - skip_mass;
    real fail1_mass   = 1 - (m_pos_above1 + m_neg_above1);

    // hit-rate for express lane
    real hit_skip = m_pos_above2 / fmax(eps, skip_mass);
    hit_skip      = fmin(1-eps, fmax(eps, hit_skip));

    if (stage[i] == 1) {
      // dropped at stage 1
      target += log(fmax(eps, fail1_mass));

    } else if (took_express[i] == 1) {       // skipped stage-2
      target += log(fmax(eps, skip_mass));
      target += (y[i]==1) ? log(hit_skip) : log1m(hit_skip);

    } else {
      target += log(fmax(eps, pass12_mass));

    {
    real eta2 = X[i, 1:last_feature_idx_to_use[2]]
                * beta_X[1:last_feature_idx_to_use[2]];
    real phi2 = inv_logit(eta2);
    phi2      = fmin(1-eps, fmax(eps, phi2));

    real log_odds_tau2_s = eta2 + log((1-tau2+eps)/(tau2+eps));
    real thr2_sig_s      = (square(deltas[2]) - 2*log_odds_tau2_s)
                            / (2*deltas[2]);

    real cdf2_pos_s  = Phi(thr2_sig_s - deltas[2]);
    real cdf2_neg_s  = Phi(thr2_sig_s);

    real m_pos_above2_s = phi2       * (1 - cdf2_pos_s);
    real m_neg_above2_s = (1 - phi2) * (1 - cdf2_neg_s);
    real search2        = m_pos_above2_s + m_neg_above2_s;
    search2             = fmin(1-eps, fmax(eps, search2));

    if (stage[i] == 2) {                // failed at stage-2
        target += log1m(search2); 
        real m_pos_below2 = phi2       * cdf2_pos_s;
        real m_neg_below2 = (1 - phi2) * cdf2_neg_s;
        m_pos_below2 = fmax(eps, m_pos_below2);
        m_neg_below2 = fmax(eps, m_neg_below2);
        //target += (y[i] == 1)
        //        ? log(m_pos_below2)
        //        : log(m_neg_below2);

    }
 else {
        target += log(search2);
        real hit_final = m_pos_above2_s / search2;
        hit_final      = fmin(1-eps, fmax(eps, hit_final));
        target += (y[i]==1) ? log(hit_final) : log1m(hit_final);
    }
    } 

    }
  } 
}


generated quantities {

  array[K] vector[M] risk_pred;              
  array[K] vector[M] search_rate_pred;       // P(pass stage k)
  vector[M]          search_rate_express_pred;
  vector[M]          hit_rate_pred;          
  vector[M]          hit_rate_stage2_pred;   // P(Y=1 | 1→2 discharge) 
  vector[M]          hit_rate_skip_pred;     // P(Y=1 | 1→3 express)    
  vector[M]          hit_rate_stage3_pred;   // P(Y=1 | 1→2→3 pass)     
  vector[M]          skip_prob_pred;         // P(1→3 | cleared stage-1)

  int n_rep_pass2                = 0;
  int n_rep_pass3                = 0;
  int n_rep_pass3_hit            = 0;
  int n_rep_pass3_total          = 0;
  int n_rep_skip                 = 0;
  int n_rep_skip_hit             = 0;   
  int n_rep_drop2_total          = 0;   
  int n_rep_drop2_hit            = 0;   
  int n_rep_stage12_normal_total = 0;   
  int n_rep_stage23_normal       = 0;   
  int n_rep_stage23_normal_hit   = 0;  
  real prop12_search_rate;
  real prop23_search_rate;

  {
    real eps = 1e-10;

    for (i in 1:M) {

      real eta1 = X[i, 1:last_feature_idx_to_use[1]]
                * beta_X[1:last_feature_idx_to_use[1]];
      real phi1 = inv_logit(eta1);
      phi1      = fmin(1 - eps, fmax(eps, phi1));
      risk_pred[1][i] = phi1;

      real tau1 = fmin(1 - eps, fmax(thresholds_in_p_space[1], eps));
      real tau2 = fmin(1 - eps, fmax(thresholds_in_p_space[2], eps));

      real log_odds_tau1 = eta1 + log((1 - tau1 + eps) / (tau1 + eps));
      real log_odds_tau2 = eta1 + log((1 - tau2 + eps) / (tau2 + eps));

      real thr1_sig = (square(deltas[1]) - 2 * log_odds_tau1) / (2 * deltas[1]);
      real thr2_sig = (square(deltas[1]) - 2 * log_odds_tau2) / (2 * deltas[1]);

      real cdf1_pos = Phi(thr1_sig - deltas[1]);
      real cdf1_neg = Phi(thr1_sig);
      real cdf2_pos = Phi(thr2_sig - deltas[1]);
      real cdf2_neg = Phi(thr2_sig);

      real m_pos_above1 = phi1       * (1 - cdf1_pos);
      real m_neg_above1 = (1 - phi1) * (1 - cdf1_neg);
      real m_pos_above2 = phi1       * (1 - cdf2_pos);
      real m_neg_above2 = (1 - phi1) * (1 - cdf2_neg);

      real skip_mass   = m_pos_above2 + m_neg_above2;                
      real pass12_mass = (m_pos_above1 + m_neg_above1) - skip_mass; 
      real fail1_mass  = 1 - (m_pos_above1 + m_neg_above1);

      search_rate_pred[1][i]     = 1 - fail1_mass;                         
      skip_prob_pred[i]          = skip_mass / (skip_mass + pass12_mass + eps); 
      search_rate_express_pred[i]= fmin(1 - eps, fmax(eps, skip_mass));    

      {
        real hit_skip = m_pos_above2 / fmax(eps, skip_mass);
        hit_rate_skip_pred[i] = fmin(1 - eps, fmax(eps, hit_skip));
      }

      real eta2 = X[i, 1:last_feature_idx_to_use[2]]
                * beta_X[1:last_feature_idx_to_use[2]];
      real phi2 = inv_logit(eta2);
      phi2      = fmin(1 - eps, fmax(eps, phi2));
      risk_pred[2][i] = phi2;

      real log_odds_tau2_s = eta2 + log((1 - tau2 + eps) / (tau2 + eps));
      real thr2_sig_s      = (square(deltas[2]) - 2 * log_odds_tau2_s)
                             / (2 * deltas[2]);

      real cdf2_pos_s = Phi(thr2_sig_s - deltas[2]);
      real cdf2_neg_s = Phi(thr2_sig_s);

      real m_pos_above2_s = phi2       * (1 - cdf2_pos_s);  
      real m_neg_above2_s = (1 - phi2) * (1 - cdf2_neg_s);
      real search2        = m_pos_above2_s + m_neg_above2_s;
      search2             = fmin(1 - eps, fmax(eps, search2));
      search_rate_pred[2][i] = search2;

      {
        real m_pos_below2_s = phi2 * cdf2_pos_s;
        real denom_b2_s     = m_pos_below2_s + (1 - phi2) * cdf2_neg_s + eps;
        hit_rate_stage2_pred[i] = fmin(1 - eps, fmax(eps, m_pos_below2_s / denom_b2_s));
      }

      {
        real hit_final = m_pos_above2_s / search2;
        hit_rate_stage3_pred[i] = fmin(1 - eps, fmax(eps, hit_final));
      }

      {
        real denom_clear = skip_mass + pass12_mass * search2 + eps;
        hit_rate_pred[i] =
          (skip_mass * hit_rate_skip_pred[i] + pass12_mass * search2 * hit_rate_stage3_pred[i])
          / denom_clear;
      }

      {
        int  stage_rep = 1;
        int  y_rep     = -1;
        real u         = uniform_rng(0, 1);

        if (u < fail1_mass) {                       
          stage_rep = 1;

        } else if (u < fail1_mass + pass12_mass) {  
          n_rep_stage12_normal_total += 1;
          if (bernoulli_rng(search2)) {             // pass stage-2 
            n_rep_stage23_normal += 1;
            stage_rep = 3;
            y_rep     = bernoulli_rng(hit_rate_stage3_pred[i]) ? 1 : 0;
            if (y_rep == 1) n_rep_stage23_normal_hit += 1;
          } else {                                  // fail stage-2 
            stage_rep = 2;
            y_rep     = bernoulli_rng(hit_rate_stage2_pred[i]) ? 1 : 0;
            n_rep_drop2_total += 1;
            if (y_rep == 1) n_rep_drop2_hit += 1;
          }

        } else {                                     // express lane 
          stage_rep = 3;
          y_rep     = bernoulli_rng(hit_rate_skip_pred[i]) ? 1 : 0;
          n_rep_skip += 1;
          if (y_rep == 1) n_rep_skip_hit += 1;
        }

        if (stage_rep >= 2) n_rep_pass2 += 1;
        if (stage_rep == 3) {
          n_rep_pass3       += 1;
          n_rep_pass3_total += 1;
          if (y_rep == 1) n_rep_pass3_hit += 1;
        }
      }
    } 
  }

  real prop_pass2_rep     = n_rep_pass2 / (1.0 * M);
  real prop_pass3_rep     = n_rep_pass3 / (1.0 * M);
  real prop_express_rep   = n_rep_skip  / (1.0 * M);

  prop12_search_rate = n_rep_stage12_normal_total / (1.0 * M);
  prop23_search_rate = n_rep_stage12_normal_total > 0
                       ? n_rep_stage23_normal / (1.0 * n_rep_stage12_normal_total)
                       : 0;

  real hit_rate_rep        = n_rep_pass3_total > 0
                             ? n_rep_pass3_hit / (1.0 * n_rep_pass3_total)
                             : 0;

  real hit_rate_stage2_rep = n_rep_drop2_total > 0
                             ? n_rep_drop2_hit / (1.0 * n_rep_drop2_total)
                             : 0;

  real hit_rate_skip_rep   = n_rep_skip > 0
                             ? n_rep_skip_hit / (1.0 * n_rep_skip)
                             : 0;

  real hit_rate_stage3_rep = n_rep_stage23_normal > 0
                             ? n_rep_stage23_normal_hit / (1.0 * n_rep_stage23_normal)
                             : 0;
}
