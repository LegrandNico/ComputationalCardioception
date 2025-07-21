# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import gc
import pandas as pd
import pymc as pm
from pathlib import Path
import numpy as np
from models import bayesian_psychophysics, shared_noise, weighted_update
import arviz as az

# load and filter the data -------------------------------------------------------------
hrd_path = Path.cwd() / "data" / "hrd.csv"
hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
hrd_df = hrd_df[~((hrd_df.cohort == "vmp1") & (hrd_df.task == "hrd-session1"))]
hrd_df = hrd_df[hrd_df.participant_id.isin(hrd_df.participant_id.unique()[:2])]

# extract variables for the model ------------------------------------------------------
n = hrd_df.participant_id.nunique()

participant_codes_intero = pd.Categorical(
    hrd_df[hrd_df.Modality == "Intero"].participant_id
).codes
participant_codes_intero_wrap = np.array(
    [np.where(participant_codes_intero == c)[0][0] for c in range(n)]
)

participant_codes_extero = pd.Categorical(
    hrd_df[hrd_df.Modality == "Extero"].participant_id
).codes

heart_rate = hrd_df[hrd_df.Modality == "Intero"].listenBPM.to_numpy()
intero_tone_2 = hrd_df[hrd_df.Modality == "Intero"].responseBPM.to_numpy()
intero_decision = (
    hrd_df[hrd_df.Modality == "Intero"].Decision.to_numpy() == "More"
).astype(int)

extero_tone_1 = hrd_df[hrd_df.Modality == "Extero"].listenBPM.to_numpy()
extero_tone_2 = hrd_df[hrd_df.Modality == "Extero"].responseBPM.to_numpy()
extero_decision = (
    hrd_df[hrd_df.Modality == "Extero"].Decision.to_numpy() == "More"
).astype(int)

summary_df = pd.DataFrame({"participant_id": hrd_df.participant_id.unique()})

# 1 - Bayesian psychophysics model -----------------------------------------------------
idata_bayesian_psychophysics, bayesian_psychophysics_model = bayesian_psychophysics(
    heart_rate=heart_rate,
    intero_tone_2=intero_tone_2,
    extero_tone_1=extero_tone_1,
    extero_tone_2=extero_tone_2,
    extero_decision=extero_decision,
    intero_decision=intero_decision,
    n=n,
    participant_codes_extero=participant_codes_extero,
    participant_codes_intero=participant_codes_intero,
)

# compute the log-likelihood
pm.compute_log_likelihood(
    idata_bayesian_psychophysics,
    model=bayesian_psychophysics_model,
)

# only keep the variables of interest
vars_to_keep = ["intero_threshold", "intero_slope", "extero_threshold", "extero_slope"]
idata_bayesian_psychophysics.posterior = idata_bayesian_psychophysics.posterior[
    vars_to_keep
]

# save the samples
az.to_netcdf(
    idata_bayesian_psychophysics, Path().cwd() / "results" / "standard_model.nc"
)

# add the parameters to the summary dataframe
summary_df["bp_intero_threshold"] = az.summary(
    idata_bayesian_psychophysics, var_names=["intero_threshold"]
)["mean"].to_list()
summary_df["bp_extero_threshold"] = az.summary(
    idata_bayesian_psychophysics, var_names=["extero_threshold"]
)["mean"].to_list()
summary_df["bp_intero_slope"] = az.summary(
    idata_bayesian_psychophysics, var_names=["intero_slope"]
)["mean"].to_list()
summary_df["bp_intero_slope"] = az.summary(
    idata_bayesian_psychophysics, var_names=["intero_slope"]
)["mean"].to_list()

# clear memory
del idata_bayesian_psychophysics
gc.collect()

# 2 - Shared noise model ---------------------------------------------------------------
idata_shared_perceptive_noise, shared_perceptive_noise = shared_noise(
    heart_rate=heart_rate,
    intero_tone_2=intero_tone_2,
    extero_tone_1=extero_tone_1,
    extero_tone_2=extero_tone_2,
    extero_decision=extero_decision,
    intero_decision=intero_decision,
    n=n,
    participant_codes_extero=participant_codes_extero,
    participant_codes_intero=participant_codes_intero,
)

# compute the log-likelihood
pm.compute_log_likelihood(idata_shared_perceptive_noise, model=shared_perceptive_noise)

# only keep the variables of interest
vars_to_keep = ["intero_threshold", "intero_slope", "extero_threshold", "extero_slope"]
idata_shared_perceptive_noise.posterior = idata_shared_perceptive_noise.posterior[
    vars_to_keep
]

# save the samples
az.to_netcdf(
    idata_shared_perceptive_noise,
    Path().cwd() / "results" / "perceptive_noise.nc",
)

# add the parameters to the summary dataframe
summary_df["sn_intero_threshold"] = az.summary(
    idata_shared_perceptive_noise, var_names=["intero_threshold"]
)["mean"].to_list()
summary_df["sn_extero_threshold"] = az.summary(
    idata_shared_perceptive_noise, var_names=["extero_threshold"]
)["mean"].to_list()
summary_df["sn_intero_slope"] = az.summary(
    idata_shared_perceptive_noise, var_names=["intero_slope"]
)["mean"].to_list()
summary_df["sn_extero_slope"] = az.summary(
    idata_shared_perceptive_noise, var_names=["extero_slope"]
)["mean"].to_list()

# clear memory
del idata_shared_perceptive_noise
gc.collect()

# 3 - Weighted update model ------------------------------------------------------------
idata_weighted_update, weigthed_update_model = weighted_update(
    heart_rate=heart_rate,
    intero_tone_2=intero_tone_2,
    extero_tone_1=extero_tone_1,
    extero_tone_2=extero_tone_2,
    extero_decision=extero_decision,
    intero_decision=intero_decision,
    n=n,
    participant_codes_extero=participant_codes_extero,
    participant_codes_intero=participant_codes_intero,
)

# compute the log-likelihood
pm.compute_log_likelihood(idata_weighted_update, model=weigthed_update_model)

# only keep the variables of interest
vars_to_keep = [
    "sensitivity",
    "mu_cardiac_prior",
    "sigma_cardiac_prior",
    "pi_cardiac_belief",
]
idata_weighted_update.posterior = idata_weighted_update.posterior[vars_to_keep]

# save the samples
az.to_netcdf(idata_weighted_update, Path().cwd() / "results" / "weighted_update.nc")

# add the parameters to the summary dataframe
summary_df["wu_sensitivity"] = az.summary(
    idata_weighted_update, var_names=["sensitivity"]
)["mean"].to_list()
summary_df["wu_mu_cardiac_prior"] = az.summary(
    idata_weighted_update, var_names=["mu_cardiac_prior"]
)["mean"].to_list()
summary_df["wu_sigma_cardiac_prior"] = az.summary(
    idata_weighted_update, var_names=["sigma_cardiac_prior"]
)["mean"].to_list()
summary_df["wu_sigma_cardiac_prior"] = az.summary(
    idata_weighted_update, var_names=["sigma_cardiac_prior"]
)["mean"].to_list()
summary_df["wu_pi_cardiac_belief"] = az.summary(
    idata_weighted_update, var_names=["pi_cardiac_belief"]
)["mean"].to_list()

# clear memory
del idata_weighted_update
gc.collect()

# save the summary dataframe
summary_df.to_csv(Path().cwd() / "results" / "summary_df.csv")
