# funnel_model
Code for ML4H 2025 paper _A Bayesian Model for Multi-stage Censoring_.

## Overview
This repository contains code for the paper "A Bayesian Model for Multi-stage Censoring" (ML4H 2025). We build a Bayesian model for settings where ground truth labels (e.g., clinical diagnosis such as cancer) are censored as a result of human decisions (e.g., whether to provide a biopsy). We specifically study the setting of multi-staged censoring (i.e., censoring that occurs because of a sequence of decisions, instead of just one). Further details on our model and setup are in our paper.

We provide an example quickstart with our model on synthetic data. We also provide scripts to clean data from [MIMIC](https://mimic.mit.edu/docs/iv/modules/ed/), an EHR database from a Boston-area hospital network, and then run our model to study emergency department decisions (e.g., whether to admit a patient to hospital, ICU, or discharge).


## Quick Start
Our model is built in Stan, and we use the `Cmdstanpy` python interface to interface with it. To install Stan, please visit [this website](https://mc-stan.org/install/). For GPU support, you'll need to install OpenCL. This is optional, and further details on Stan with OpenCL are available [here](https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html).

Additional required packages are listed in `requirements.txt`. To build the environment, we'd recommend first installing Stan + OpenCL then running `pip install -r requirements` in a new environment.

## Synthetic Example
We provide an example input to the funnel model in example_data.csv. You can run the funnel model on this data by running `python synthetic_example.py`. This uses the model provided in `models/funnel_model.stan`, which is the most general model we provide (i.e., can support arbitrary numbers of stages). Although this model is general, we emphasize that we have not tested for large numbers of stages (i.e., K > 3 total stages).

To use your own data, you can follow the data format in `example_data.csv`. You may use the functions in `utils.py` to generate synthetic data. The function `generate_chain_data` accepts model parameters (i.e., thresholds, $\delta$s, $\beta$s) and raw data and simulates patient flows through the funnel model.

Once you have your data, `synthetic_example.py` requires a few arguments:

* `M`: the total number of patients/individuals in the dataset.
* `N`: the total number of features.
* `K`: the total number of stages.
* `last_feature_idx_to_use`: the a list of the last feature available at each stage.
* `samples`: the number of sampling steps to use in MCMC.
* `warmup`: the number of warmup steps to use in MCMC.

In addition, we recommend normalizing all features to be centered at 0 with standard deviation of 1 (i.e. z-scoring all features). We have noticed that using unnormalized feature values can result in poor convergence. 

To assess the posterior fit, we recommend checking the values of [Rhat](https://mc-stan.org/rstan/reference/Rhat.html) provided by Stan. In general, Rhat values should all be below 1.1. Deviations from this often indicate poor convergence. You can also check the [traceplots](https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html) to ensure that the MCMC chains (visually) mix well. Additionally, we provide posterior predictive values in the `generated_quantities` block of the Stan models. These values should roughly match the true data (e.g., posterior on number of patients who make it past the first stage should match the true data).

## MIMIC
We also provide scripts to recreate the gender disparity results from our paper. `data_cleaning/extract_main_dataset.ipynb` provides code to clean and format the raw MIMIC data. We are unable to provide the cleaned data itself, as MIMIC is a private medical dataset. To apply for access, see [here](https://mimic.mit.edu/docs/help/).

Once you have the cleaned data, you can run `python mimic_model.py --model original --group_on gender --group_on_value F --data_type staged_quadratic --path_to_base_results_dir [PATH] --path_to_mimic_data [PATH]`. By default, this runs the model on the entire MIMIC dataset among female patients only. If you choose to run on a (random) subset, you can use `--sample_size 5000` or similar. A results directory will be created in the path specified in `--path_to_base_results_dir` if it doesn't already exist.
