# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from typing import Union


def extract_psychometric(
    intensities: np.ndarray, decision: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the x, r, and n vectors from response data.

    Parameters
    ----------
    intensities :
        1D array of stimulus intensities.
    decision :
        1D array of binary decisions (1 for 'faster', 0 for 'slower').

    Returns
    -------
    x : np.ndarray
        Unique stimulus intensities.
    n : np.ndarray
        Number of trials for each intensity.
    r : np.ndarray
        Number of 'faster' responses for each intensity.

    """
    n_items = len(np.unique(intensities))
    x, n, r = np.zeros(n_items), np.zeros(n_items), np.zeros(n_items)

    for ii, intensity in enumerate(np.unique(intensities)):
        x[ii] = intensity
        n[ii] = sum(intensities == intensity)
        r[ii] = sum((intensities == intensity) & (decision == 1))

    return x, n, r


def cumulative_normal(
    x: Union[float, np.ndarray, TensorVariable],
    threshold: Union[float, np.ndarray, TensorVariable],
    slope: Union[float, np.ndarray, TensorVariable],
) -> np.ndarray:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 + 0.5 * pt.erf((x - threshold) / (slope * pt.sqrt(2.0)))


def weighted_bayesian_update_mean(
    mu_0: Union[float, np.ndarray, TensorVariable],
    mu_1: Union[float, np.ndarray, TensorVariable],
    pi_0: Union[float, np.ndarray, TensorVariable],
    pi_1: Union[float, np.ndarray, TensorVariable],
    pi: Union[float, np.ndarray, TensorVariable],
    lam: Union[float, np.ndarray, TensorVariable],
):
    """Weighted Bayesian update for the mean of a distribution."""
    mu = ((1 - lam) * mu_0 * pi_0 + lam * mu_1 * pi_1) / pi
    return mu


def weighted_bayesian_update_precision(
    pi_0: Union[float, np.ndarray, TensorVariable],
    pi_1: Union[float, np.ndarray, TensorVariable],
    lam: Union[float, np.ndarray, TensorVariable],
):
    """Weighted Bayesian update for the precision of a distribution."""
    pi = (1 - lam) * pi_0 + lam * pi_1
    return pi
