import numpy as np
import pytensor.tensor as pt


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


def cumulative_normal(x: float, threshold: float, slope: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 + 0.5 * pt.erf((x - threshold) / (slope * pt.sqrt(2.0)))
