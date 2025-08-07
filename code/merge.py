import arviz as az
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

hrd_path = Path.cwd() / "data" / "hrd.csv"
hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
participants_list = hrd_df.participant_id.unique()

# Initialize an empty DataFrame to store the summary
cardiac_hgf_df = pd.DataFrame()

for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        # Cardiac HGF model ------------------------------------------------------------
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
            pi_1 = 1.0 / (1.0 + np.exp(exteroceptive_tonic_volatility))
            exteroceptive_sensitivity = exteroceptive_precision / (
                exteroceptive_precision + pi_1
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
                        }
                    ),
                ],
                ignore_index=True,
            )


cardiac_hgf_df.to_csv(Path.cwd() / "results" / "cardiac_hgf_summary.csv", index=False)
print("Completed all models and saved the summary dataframe.")


#     # add the parameters to the summary dataframe
#     summary_df["bp_intero_threshold"] = az.summary(
#         idata_bayesian_psychophysics, var_names=["intero_threshold"]
#     )["mean"].to_list()
#     summary_df["bp_extero_threshold"] = az.summary(
#         idata_bayesian_psychophysics, var_names=["extero_threshold"]
#     )["mean"].to_list()
#     summary_df["bp_intero_slope"] = az.summary(
#         idata_bayesian_psychophysics, var_names=["intero_slope"]
#     )["mean"].to_list()
#     summary_df["bp_intero_slope"] = az.summary(
#         idata_bayesian_psychophysics, var_names=["intero_slope"]
#     )["mean"].to_list()

#     # add the parameters to the summary dataframe
#     summary_df["sn_intero_threshold"] = az.summary(
#         idata_shared_perceptive_noise, var_names=["intero_threshold"]
#     )["mean"].to_list()
#     summary_df["sn_extero_threshold"] = az.summary(
#         idata_shared_perceptive_noise, var_names=["extero_threshold"]
#     )["mean"].to_list()
#     summary_df["sn_intero_slope"] = az.summary(
#         idata_shared_perceptive_noise, var_names=["intero_slope"]
#     )["mean"].to_list()
#     summary_df["sn_extero_slope"] = az.summary(
#         idata_shared_perceptive_noise, var_names=["extero_slope"]
#     )["mean"].to_list()

#     # add the parameters to the summary dataframe
#     summary_df["wu_sensitivity"] = az.summary(
#         idata_weighted_update, var_names=["sensitivity"]
#     )["mean"].to_list()
#     summary_df["wu_mu_cardiac_prior"] = az.summary(
#         idata_weighted_update, var_names=["mu_cardiac_prior"]
#     )["mean"].to_list()
#     summary_df["wu_sigma_cardiac_prior"] = az.summary(
#         idata_weighted_update, var_names=["sigma_cardiac_prior"]
#     )["mean"].to_list()
#     summary_df["wu_sigma_cardiac_prior"] = az.summary(
#         idata_weighted_update, var_names=["sigma_cardiac_prior"]
#     )["mean"].to_list()
#     summary_df["wu_pi_cardiac_belief"] = az.summary(
#         idata_weighted_update, var_names=["pi_cardiac_belief"]
#     )["mean"].to_list()

#         summary_df = pd.DataFrame({"participant_id": hrd_df.participant_id.unique()})

# # save the summary dataframe
# summary_df.to_csv(Path().cwd() / "results" / f"summary_df_del_{args.session}.csv")
