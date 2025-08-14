# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pytensor

pytensor.config.mode == "NUMBA"
import pymc as pm
import numpy as np
from models.utils import cumulative_normal
import pytensor.tensor as pt


def bayesian_psychophysics(
    heart_rate: np.ndarray,
    intero_tone_2: np.ndarray,
    extero_tone_1: np.ndarray,
    extero_tone_2: np.ndarray,
    extero_decision: np.ndarray,
    intero_decision: np.ndarray,
    n: int,
    participant_codes_extero: np.ndarray,
    participant_codes_intero: np.ndarray,
):
    """Bayesian psychophysics model of cardiac interoception."""
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

        # interoception ------------------------------
        intero_threshold = pm.Uniform(
            "intero_threshold", lower=-50, upper=50, shape=(n,)
        )
        intero_slope = pm.Uniform("intero_slope", lower=0.1, upper=30, shape=(n,))
        theta_intero = pm.Deterministic(
            "theta_intero",
            cumulative_normal(
                intero_tone_2 - heart_rate,
                intero_threshold[participant_codes_intero],
                intero_slope[participant_codes_intero],
            ),
        )

        thetas = pm.Deterministic(
            "thetas", pt.concatenate([theta_extero, theta_intero])
        )
        _ = pm.Binomial(
            "bin",
            p=thetas,
            n=1,
            observed=np.append(extero_decision, intero_decision),
        )

        idata = pm.sample(
            chains=4,
            cores=1,
            draws=1000,
            return_inferencedata=True,
            nuts_sampler="nutpie",
        )

    return idata, model
