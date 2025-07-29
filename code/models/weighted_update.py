# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pytensor

pytensor.config.mode == "NUMBA"
import pymc as pm
import numpy as np
import pytensor.tensor as pt
from models.utils import (
    cumulative_normal,
    weighted_bayesian_update_precision,
    weighted_bayesian_update_mean,
)


def weighted_update(
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
    """Weighted Bayesian update for interoception trials."""
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
                pt.sqrt(2 * extero_slope[participant_codes_extero] ** 2),
            ),
        )
        _ = pm.Binomial("bin_extero", p=theta_extero, n=1, observed=extero_decision)

        # interoception ------------------------------

        # priors over cardiac beliefs
        mu_cardiac_prior = pm.Uniform(
            "mu_cardiac_prior", lower=20, upper=150, shape=(n,)
        )
        sigma_cardiac_prior = pm.Uniform(
            "sigma_cardiac_prior", lower=0.1, upper=30, shape=(n,)
        )

        interoceptive_precision = 1.0
        w = pm.Beta("w", 1, 1, shape=(n,))

        # update cardiac beliefs using physiological inputs
        pi_cardiac_belief = pm.Deterministic(
            "pi_cardiac_belief",
            weighted_bayesian_update_precision(
                pi_0=1 / (sigma_cardiac_prior**2),
                pi_1=interoceptive_precision,
                lam=w,
            ),
        )

        mus_cardiac_belief = pm.Deterministic(
            "mus_cardiac_belief",
            weighted_bayesian_update_mean(
                mu_0=mu_cardiac_prior[participant_codes_intero],
                mu_1=heart_rate,
                pi_0=1 / (sigma_cardiac_prior[participant_codes_intero] ** 2),
                pi_1=interoceptive_precision,
                pi=pi_cardiac_belief[participant_codes_intero],
                lam=w[participant_codes_intero],
            ),
        )

        _ = pm.Deterministic(
            "sensitivity",
            (interoceptive_precision * w) / pi_cardiac_belief,
        )

        theta_intero = pm.Deterministic(
            "theta_intero",
            1.0
            - cumulative_normal(
                0.0,
                intero_tone_2 - mus_cardiac_belief,
                pt.sqrt(
                    (1 / pi_cardiac_belief[participant_codes_intero]) ** 2
                    + extero_slope[participant_codes_intero] ** 2
                ),
            ),
        )
        _ = pm.Binomial("bin_intero", p=theta_intero, n=1, observed=intero_decision)

        idata = pm.sample(
            chains=4,
            cores=4,
            draws=1000,
            return_inferencedata=True,
            nuts_sampler="nutpie",
        )

    return idata, model
