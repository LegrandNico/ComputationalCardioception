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
        # exteroception ----------------------------------------------------------------

        # priors over cardiac beliefs
        mu_extero_prior = pm.Uniform("mu_extero_prior", lower=20, upper=150, shape=(n,))
        sigma_extero_prior = pm.Uniform(
            "sigma_extero_prior", lower=2.0, upper=30, shape=(n,)
        )

        extero_std = pm.Uniform("extero_std", lower=2.0, upper=30, shape=(n,))
        omega_extero = pm.Beta("omega_extero", 1, 1, shape=(n,))

        # update cardiac beliefs using physiological inputs
        pi_extero_belief = pm.Deterministic(
            "pi_extero_belief",
            weighted_bayesian_update_precision(
                pi_0=1 / (sigma_extero_prior**2),
                pi_1=1 / (extero_std**2),
                omega=omega_extero,
            ),
        )

        mus_extero_belief = pm.Deterministic(
            "mus_extero_belief",
            weighted_bayesian_update_mean(
                mu_0=mu_extero_prior[participant_codes_extero],
                mu_1=extero_tone_1,
                pi_0=1 / (sigma_extero_prior[participant_codes_extero] ** 2),
                pi_1=1 / (extero_std**2),
                pi=pi_extero_belief[participant_codes_extero],
                omega=omega_extero[participant_codes_extero],
            ),
        )

        _ = pm.Deterministic(
            "extero_sensitivity",
            (omega_extero * (1 / (extero_std**2))) / pi_extero_belief,
        )

        theta_extero = pm.Deterministic(
            "theta_extero",
            1.0
            - cumulative_normal(
                0.0,
                extero_tone_2 - mus_extero_belief,
                pt.sqrt(
                    (1 / pi_extero_belief[participant_codes_extero])
                    + extero_std[participant_codes_extero] ** 2
                ),
            ),
        )
        
        _ = pm.Deterministic("auditory_belief", pt.mean(mus_extero_belief))

        _ = pm.Binomial(
            "bin_extero",
            p=theta_extero,
            n=1,
            observed=extero_decision,
        )

        # interoception ----------------------------------------------------------------

        # priors over cardiac beliefs
        mu_cardiac_prior = pm.Uniform(
            "mu_cardiac_prior", lower=20, upper=150, shape=(n,)
        )
        sigma_cardiac_prior = pm.Uniform(
            "sigma_cardiac_prior", lower=2.0, upper=30, shape=(n,)
        )

        interoceptive_precision = pm.Uniform(
            "interoceptive_precision", lower=0.01, upper=1.0, shape=(n,)
        )
        omega_intero = pm.Beta("omega_intero", 1, 1, shape=(n,))

        # update cardiac beliefs using physiological inputs
        pi_cardiac_belief = pm.Deterministic(
            "pi_cardiac_belief",
            weighted_bayesian_update_precision(
                pi_0=1 / (sigma_cardiac_prior**2),
                pi_1=interoceptive_precision,
                omega=omega_intero,
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
                omega=omega_intero[participant_codes_intero],
            ),
        )

        _ = pm.Deterministic(
            "intero_sensitivity",
            (omega_intero * interoceptive_precision) / pi_cardiac_belief,
        )

        theta_intero = pm.Deterministic(
            "theta_intero",
            1.0
            - cumulative_normal(
                0.0,
                intero_tone_2 - mus_cardiac_belief,
                pt.sqrt(
                    (1 / pi_cardiac_belief[participant_codes_intero])
                    + extero_std[participant_codes_intero] ** 2
                ),
            ),
        )
        
        _ = pm.Deterministic("cardiac_belief", pt.mean(mus_cardiac_belief))

        _ = pm.Binomial(
            "bin_intero",
            p=theta_intero,
            n=1,
            observed=intero_decision,
        )

        idata = pm.sample(
            chains=4,
            cores=4,
            draws=1000,
            return_inferencedata=True,
            nuts_sampler="nutpie",
        )

    return idata, model
