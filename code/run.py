# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import gc
import pandas as pd
import pymc as pm
import numpy as np
from pathlib import Path
from models import (
    bayesian_psychophysics,
    cardiac_believing,
    weighted_update,
    sample_cardiac_hgf,
)
import arviz as az
import argparse
import multiprocessing as mp
from functools import partial
from jax import config

config.update("jax_enable_x64", True)


def individual_fit(participant_id: str, session: int, model: str, overwrite: bool):
    """Fit the models for each individual participant."""
    hrd_path = Path.cwd() / "data" / "hrd.csv"
    hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
    if session == 1:
        hrd_df = hrd_df[
            (
                (hrd_df.cohort == "vmp1")
                & (hrd_df.task == "hrd-session1")
                & (hrd_df.participant_id == participant_id)
                & (~hrd_df.Decision.isnull())
            )
        ]
    elif session == 2:
        hrd_df = hrd_df[
            (
                ((hrd_df.cohort == "vmp1") & (hrd_df.task == "hrd-session2"))
                | ((hrd_df.cohort == "vmp2") & (hrd_df.task == "hrd-session1"))
            )
            & (hrd_df.participant_id == participant_id)
            & (~hrd_df.Decision.isnull())
        ]

    # extract variables for the model --------------------------------------------------
    n = hrd_df.participant_id.nunique()
    if (n != 1) or (len(hrd_df) < 20) or (len(hrd_df) > 160):
        return

    participant_codes_intero = pd.Categorical(
        hrd_df[hrd_df.Modality == "Intero"].participant_id
    ).codes

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

    # 1 - Bayesian psychophysics model -------------------------------------------------
    if model == "all" or model == "bayesian_psychophysics":
        if (
            Path().cwd()
            / "results"
            / "idata"
            / f"standard_model_session{session}_{participant_id}.nc"
        ).exists() and not overwrite:
            print(f"Standard model already exists, for {participant_id} skipping.")
            return
        idata_bayesian_psychophysics, bayesian_psychophysics_model = (
            bayesian_psychophysics(
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
        )

        # only keep the variables of interest
        vars_to_keep = [
            "intero_threshold",
            "intero_slope",
            "extero_threshold",
            "extero_slope",
        ]
        idata_bayesian_psychophysics.posterior = idata_bayesian_psychophysics.posterior[
            vars_to_keep
        ]

        # compute the log-likelihood
        pm.compute_log_likelihood(
            idata_bayesian_psychophysics,
            model=bayesian_psychophysics_model,
            var_names=["bin"],
        )

        # save the samples
        az.to_netcdf(
            az.InferenceData(
                posterior=idata_bayesian_psychophysics.posterior,
                log_likelihood=idata_bayesian_psychophysics.log_likelihood,
            ),
            Path().cwd()
            / "results"
            / "idata"
            / f"standard_model_session{session}_{participant_id}.nc",
        )

        # clear memory
        del idata_bayesian_psychophysics
        gc.collect()

    # 2 - Cardiac believing-------------------------------------------------------------
    if model == "all" or model == "cardiac_believing":
        if (
            Path().cwd()
            / "results"
            / "idata"
            / f"cardiac_believing_session{session}_{participant_id}.nc"
        ).exists() and not overwrite:
            print(
                f"Cardiac believing model already exists, for {participant_id} skipping."
            )
            return
        idata_cardiac_believing, shared_perceptive_noise = cardiac_believing(
            intero_tone_2=intero_tone_2,
            extero_tone_1=extero_tone_1,
            extero_tone_2=extero_tone_2,
            extero_decision=extero_decision,
            intero_decision=intero_decision,
            n=n,
            participant_codes_extero=participant_codes_extero,
            participant_codes_intero=participant_codes_intero,
        )

        # only keep the variables of interest
        vars_to_keep = [
            "intero_mean",
            "intero_std",
            "extero_threshold",
            "extero_slope",
        ]
        idata_cardiac_believing.posterior = idata_cardiac_believing.posterior[
            vars_to_keep
        ]
        # compute the log-likelihood
        pm.compute_log_likelihood(
            idata_cardiac_believing,
            model=shared_perceptive_noise,
            var_names=["bin"],
        )

        # save the samples
        az.to_netcdf(
            az.InferenceData(
                posterior=idata_cardiac_believing.posterior,
                log_likelihood=idata_cardiac_believing.log_likelihood,
            ),
            Path().cwd()
            / "results"
            / "idata"
            / f"cardiac_believing_session{session}_{participant_id}.nc",
        )

        # clear memory
        del idata_cardiac_believing
        gc.collect()

    # 3 - Weighted update model ------------------------------------------------------------
    if model == "all" or model == "weighted_update":
        if (
            Path().cwd()
            / "results"
            / "idata"
            / f"weighted_update_session{session}_{participant_id}.nc"
        ).exists() and not overwrite:
            print(
                f"Weighted update model already exists, for {participant_id} skipping."
            )
            return
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

        # only keep the variables of interest
        vars_to_keep = [
            "sensitivity",
            "mu_cardiac_prior",
            "sigma_cardiac_prior",
            "pi_cardiac_belief",
            "extero_std",
            "w",
        ]
        idata_weighted_update.posterior = idata_weighted_update.posterior[vars_to_keep]

        # compute the log-likelihood
        pm.compute_log_likelihood(
            idata_weighted_update, model=weigthed_update_model, var_names=["bin"]
        )

        # save the samples
        az.to_netcdf(
            az.InferenceData(
                posterior=idata_weighted_update.posterior,
                log_likelihood=idata_weighted_update.log_likelihood,
            ),
            Path().cwd()
            / "results"
            / "idata"
            / f"weighted_update_session{session}_{participant_id}.nc",
        )

        # clear memory
        del idata_weighted_update
        gc.collect()

    # 4 - The cardiac HGF --------------------------------------------------------------
    if model == "all" or model == "cardiac_hgf":
        if (
            Path().cwd()
            / "results"
            / "idata"
            / f"cardiac_hgf_session{session}_{participant_id}.nc"
        ).exists() and not overwrite:
            print(f"Cardiac HGF model already exists, for {participant_id} skipping.")
            return
        input_data_extero = np.array([extero_tone_1, extero_tone_2]).T
        input_data_intero = np.array([heart_rate, intero_tone_2]).T
        try:
            idata_cardiac_hgf, model_cardiac_hgf = sample_cardiac_hgf(
                input_data_extero=input_data_extero,
                input_data_intero=input_data_intero,
                extero_decision=extero_decision,
                intero_decision=intero_decision,
            )

            # compute the log-likelihood
            pm.compute_log_likelihood(
                idata_cardiac_hgf, model=model_cardiac_hgf, var_names=["bin"]
            )
            # save the samples
            az.to_netcdf(
                az.InferenceData(
                    posterior=idata_cardiac_hgf.posterior,
                    log_likelihood=idata_cardiac_hgf.log_likelihood,
                ),
                Path().cwd()
                / "results"
                / "idata"
                / f"cardiac_hgf_session{session}_{participant_id}.nc",
            )
        except pm.exceptions.SamplingError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True, default=1)
    parser.add_argument("--model", type=str, required=True, default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # load and filter the data -------------------------------------------------------------
    hrd_path = Path.cwd() / "data" / "hrd.csv"
    hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
    if args.session == 1:
        hrd_df = hrd_df[(hrd_df.cohort == "vmp1") & (hrd_df.task == "hrd-session1")]
    elif args.session == 2:
        hrd_df = hrd_df[
            ((hrd_df.cohort == "vmp1") & (hrd_df.task == "hrd-session2"))
            | ((hrd_df.cohort == "vmp2") & (hrd_df.task == "hrd-session1"))
        ]

    partial_fn = partial(
        individual_fit, session=args.session, model=args.model, overwrite=args.overwrite
    )
    pool = mp.Pool(processes=25)
    pool.map(partial_fn, hrd_df.participant_id.unique())
    pool.close()

    print("All reports generated successfully.")
