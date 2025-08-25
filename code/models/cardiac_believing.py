# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pytensor

pytensor.config.mode == "NUMBA"
import pymc as pm
import numpy as np
from models.utils import cumulative_normal


def cardiac_believing(
    intero_tone_2: np.ndarray,
    extero_tone_1: np.ndarray,
    extero_tone_2: np.ndarray,
    extero_decision: np.ndarray,
    intero_decision: np.ndarray,
    n: int,
    participant_codes_extero: np.ndarray,
    participant_codes_intero: np.ndarray,
):
    """Bayesian psychophysics model of cardiac believing."""
    with pm.Model() as model:
        # exteroception ------------------------------
        extero_threshold = pm.Uniform(
            "extero_threshold", lower=-50, upper=50, shape=(n,)
        )
        extero_slope = pm.Uniform("extero_slope", lower=0.1, upper=30, shape=(n,))
        theta_extero = pm.Deterministic(
            "theta_extero",
            cumulative_normal(
                extero_tone_2 - extero_tone_1,
                extero_threshold[participant_codes_extero],
                extero_slope[participant_codes_extero],
            ),
        )

        _ = pm.Binomial(
            "bin_extero",
            p=theta_extero,
            n=1,
            observed=extero_decision,
        )

        # interoception ------------------------------
        intero_mean = pm.Uniform("intero_mean", lower=5, upper=200, shape=(n,))
        intero_std = pm.Uniform("intero_std", lower=0.1, upper=60, shape=(n,))
        theta_intero = pm.Deterministic(
            "theta_intero",
            cumulative_normal(
                intero_tone_2,
                intero_mean[participant_codes_intero],
                intero_std[participant_codes_intero],
            ),
        )

        _ = pm.Binomial(
            "bin_intero",
            p=theta_intero,
            n=1,
            observed=intero_decision,
        )

        idata = pm.sample(
            chains=4,
            cores=1,
            draws=1000,
            return_inferencedata=True,
            nuts_sampler="nutpie",
        )

    return idata, model
