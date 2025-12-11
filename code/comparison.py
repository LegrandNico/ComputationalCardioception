# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # az.compare will throw warnings
plt.rcParams["figure.constrained_layout.use"] = True

hrd_path = Path.cwd() / "data" / "hrd.csv"
hrd_df = pd.read_csv(hrd_path, index_col=0, low_memory=False)
participants_list = hrd_df.participant_id.unique()

missing_participants = []
model_comparison_df = pd.DataFrame([])
for participant_id in tqdm(participants_list):
    for session in [1, 2]:
        standard_path = (
            Path().cwd()
            / "results"
            / "idata"
            / f"standard_model_session{session}_{participant_id}.nc"
        )

        weighted_update_path = (
            Path().cwd()
            / "results"
            / "idata"
            / f"weighted_update_session{session}_{participant_id}.nc"
        )

        cardiac_believing_path = (
            Path().cwd()
            / "results"
            / "idata"
            / f"cardiac_believing_session{session}_{participant_id}.nc"
        )

        cardiac_hgf_path = (
            Path().cwd()
            / "results"
            / "idata"
            / f"cardiac_hgf_session{session}_{participant_id}.nc"
        )

        if (
            standard_path.exists()
            & weighted_update_path.exists()
            & cardiac_believing_path.exists()
            & cardiac_hgf_path.exists()
        ):
            model_compare = az.compare(
                {
                    "standard": az.from_netcdf(standard_path),
                    "cardiac_beliefs": az.from_netcdf(weighted_update_path),
                    "weigthed_update": az.from_netcdf(cardiac_believing_path),
                    "cardiac_hgf": az.from_netcdf(cardiac_hgf_path),
                },
            )
            model_compare["participant_id"] = participant_id
            model_compare["session"] = session
            model_comparison_df = pd.concat([model_comparison_df, model_compare])

        else:
            missing_participants.append((participant_id, session))
