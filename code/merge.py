# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import arviz as az
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

hrd_path = Path.cwd() / "data" / "hrd.csv"
hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
participants_list = hrd_df.participant_id.unique()

# standard model -----------------------------------------------------------------------

# Initialize an empty DataFrame to store the summary
standard_model_df = pd.DataFrame()

for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        path = (
            Path.cwd()
            / "results"
            / "idata"
            / f"standard_model_session{session}_{participant_id}.nc"
        )

        if path.exists():
            idata_standard = az.from_netcdf(path)

            posterior = az.extract(idata_standard, group="posterior")
            extero_threshold = posterior["extero_threshold"].mean(dim="sample").values
            extero_slope = posterior["extero_slope"].mean(dim="sample").values
            intero_threshold = posterior["intero_threshold"].mean(dim="sample").values
            intero_slope = posterior["intero_slope"].mean(dim="sample").values

            standard_model_df = pd.concat(
                [
                    standard_model_df,
                    pd.DataFrame(
                        {
                            "participant_id": [participant_id],
                            "session": [session],
                            "extero_threshold": extero_threshold,
                            "extero_slope": extero_slope,
                            "intero_threshold": intero_threshold,
                            "intero_slope": intero_slope,
                        }
                    ),
                ],
                ignore_index=True,
            )


standard_model_df.to_csv(
    Path.cwd() / "results" / "standard_model_summary.csv", index=False
)


# Cardiac believing --------------------------------------------------------------------

# Initialize an empty DataFrame to store the summary
cardiac_believing_df = pd.DataFrame()

for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        path = (
            Path.cwd()
            / "results"
            / "idata"
            / f"cardiac_believing_session{session}_{participant_id}.nc"
        )

        if path.exists():
            idata_cardiac_believing = az.from_netcdf(path)

            posterior = az.extract(idata_cardiac_believing, group="posterior")
            intero_mean = posterior["intero_mean"].mean(dim="sample").values
            intero_std = posterior["intero_std"].mean(dim="sample").values
            extero_mean = posterior["extero_mean"].mean(dim="sample").values
            extero_std = posterior["extero_std"].mean(dim="sample").values

            cardiac_believing_df = pd.concat(
                [
                    cardiac_believing_df,
                    pd.DataFrame(
                        {
                            "participant_id": [participant_id],
                            "session": [session],
                            "intero_mean": intero_mean,
                            "intero_std": intero_std,
                            "extero_mean": extero_mean,
                            "extero_std": extero_std,
                        }
                    ),
                ],
                ignore_index=True,
            )


cardiac_believing_df.to_csv(
    Path.cwd() / "results" / "cardiac_believing_summary.csv", index=False
)

# Weighted update ----------------------------------------------------------------------

# Initialize an empty DataFrame to store the summary
weighted_update_df = pd.DataFrame()

for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        path = (
            Path.cwd()
            / "results"
            / "idata"
            / f"weighted_update_session{session}_{participant_id}.nc"
        )

        if path.exists():
            idata_weighted_update = az.from_netcdf(path)

            posterior = az.extract(idata_weighted_update, group="posterior")

            # interoception ------------------------------------------------------------
            intero_sensitivity = (
                posterior["intero_sensitivity"].mean(dim="sample").values
            )
            omega_intero = posterior["omega_intero"].mean(dim="sample").values
            pi_cardiac_belief = posterior["pi_cardiac_belief"].mean(dim="sample").values
            sigma_cardiac_prior = (
                posterior["sigma_cardiac_prior"].mean(dim="sample").values
            )
            mu_cardiac_prior = posterior["mu_cardiac_prior"].mean(dim="sample").values

            logit_interoceptive_sensitivity = np.log(
                intero_sensitivity / (1 - intero_sensitivity)
            )

            # exteroception ------------------------------------------------------------
            extero_sensitivity = (
                posterior["extero_sensitivity"].mean(dim="sample").values
            )
            omega_extero = posterior["omega_extero"].mean(dim="sample").values
            pi_extero_belief = posterior["pi_extero_belief"].mean(dim="sample").values
            sigma_extero_prior = (
                posterior["sigma_extero_prior"].mean(dim="sample").values
            )
            mu_extero_prior = posterior["mu_extero_prior"].mean(dim="sample").values

            logit_exteroceptive_sensitivity = np.log(
                extero_sensitivity / (1 - extero_sensitivity)
            )

            weighted_update_df = pd.concat(
                [
                    weighted_update_df,
                    pd.DataFrame(
                        {
                            "participant_id": [participant_id],
                            "session": [session],
                            "intero_sensitivity": intero_sensitivity,
                            "extero_sensitivity": extero_sensitivity,
                            "pi_cardiac_belief": pi_cardiac_belief,
                            "pi_extero_belief": pi_extero_belief,
                            "sigma_cardiac_prior": sigma_cardiac_prior,
                            "sigma_extero_prior": sigma_extero_prior,
                            "mu_cardiac_prior": mu_cardiac_prior,
                            "mu_extero_prior": mu_extero_prior,
                            "logit_interoceptive_sensitivity": logit_interoceptive_sensitivity,
                            "logit_exteroceptive_sensitivity": logit_exteroceptive_sensitivity,
                        }
                    ),
                ],
                ignore_index=True,
            )


weighted_update_df.to_csv(
    Path.cwd() / "results" / "weighted_update_summary.csv", index=False
)

# Cardiac HGF --------------------------------------------------------------------------

# Initialize an empty DataFrame to store the summary
cardiac_hgf_df = pd.DataFrame()

for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        path = (
            Path.cwd()
            / "results"
            / "idata"
            / f"cardiac_hgf_session{session}_{participant_id}.nc"
        )

        if path.exists():
            idata_cardiac_hgf = az.from_netcdf(path)

            posterior = az.extract(idata_cardiac_hgf, group="posterior")
            interoceptive_precision = (
                posterior["interoceptive_precision"].mean(dim="sample").values
            )
            exteroceptive_precision = (
                posterior["exteroceptive_precision"].mean(dim="sample").values
            )
            interoceptive_tonic_volatility = (
                posterior["interoceptive_tonic_volatility"].mean(dim="sample").values
            )
            exteroceptive_tonic_volatility = (
                posterior["exteroceptive_tonic_volatility"].mean(dim="sample").values
            )

            # recover the implied sensitivity by applying the hgf update
            # for prediction of the precision at the second level
            pi_1 = 1.0 / (1.0 + np.exp(interoceptive_tonic_volatility))
            interoceptive_sensitivity = interoceptive_precision / (
                interoceptive_precision + pi_1
            )

            # use the logit of the sensitivity
            logit_interoceptive_sensitivity = np.log(interoceptive_precision) - np.log(
                pi_1
            )

            pi_1 = 1.0 / (1.0 + np.exp(exteroceptive_tonic_volatility))
            exteroceptive_sensitivity = exteroceptive_precision / (
                exteroceptive_precision + pi_1
            )
            # use the logit of the sensitivity
            logit_exteroceptive_sensitivity = np.log(exteroceptive_precision) - np.log(
                pi_1
            )

            cardiac_hgf_df = pd.concat(
                [
                    cardiac_hgf_df,
                    pd.DataFrame(
                        {
                            "participant_id": [participant_id],
                            "session": [session],
                            "interoceptive_precision": interoceptive_precision,
                            "exteroceptive_precision": exteroceptive_precision,
                            "interoceptive_tonic_volatility": interoceptive_tonic_volatility,
                            "exteroceptive_tonic_volatility": exteroceptive_tonic_volatility,
                            "exteroceptive_mean": posterior["exteroceptive_mean"]
                            .mean(dim="sample")
                            .values,
                            "interoceptive_mean": posterior["interoceptive_mean"]
                            .mean(dim="sample")
                            .values,
                            "interoceptive_sensitivity": interoceptive_sensitivity,
                            "exteroceptive_sensitivity": exteroceptive_sensitivity,
                            "logit_interoceptive_sensitivity": logit_interoceptive_sensitivity,
                            "logit_exteroceptive_sensitivity": logit_exteroceptive_sensitivity,
                        }
                    ),
                ],
                ignore_index=True,
            )


cardiac_hgf_df.to_csv(Path.cwd() / "results" / "cardiac_hgf_summary.csv", index=False)


print("Completed all models and saved the summary dataframe.")
