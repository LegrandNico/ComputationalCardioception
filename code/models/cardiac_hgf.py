# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.model import Network
from jax import Array
from jax.scipy.stats.norm import cdf
import jax.numpy as jnp
import numpy as np
import pymc as pm
from pytensor import wrap_jax


def cardiac_hgf(
    interoceptive_precision: float,
    interoceptive_tonic_volatility: float,
    interoceptive_mean: float,
    exteroceptive_precision: float,
    exteroceptive_tonic_volatility: float,
    exteroceptive_mean: float,
    input_data_extero: np.ndarray,
    input_data_intero: np.ndarray,
) -> Array:
    """Fit the cardiac hierarchical Gaussian filter to single participant."""
    # create new network structures for this participant
    extero_network = (
        Network(volatility_updates="unbounded")
        .add_nodes(
            precision=exteroceptive_precision,
            expected_precision=exteroceptive_precision,
        )
        .add_nodes(
            value_children=0,
            mean=exteroceptive_mean,
            tonic_volatility=exteroceptive_tonic_volatility,
        )
        .add_nodes(
            precision=exteroceptive_precision,
            expected_precision=exteroceptive_precision,
        )
        .add_nodes(
            value_children=2,
            mean=exteroceptive_mean,
            tonic_volatility=exteroceptive_tonic_volatility,
        )
    )

    intero_network = (
        Network(volatility_updates="unbounded")
        .add_nodes(
            precision=interoceptive_precision,
            expected_precision=interoceptive_precision,
        )
        .add_nodes(
            value_children=0,
            mean=interoceptive_mean,
            tonic_volatility=interoceptive_tonic_volatility,
        )
        .add_nodes(
            precision=exteroceptive_precision,
            expected_precision=exteroceptive_precision,
        )
        .add_nodes(
            value_children=2,
            mean=exteroceptive_mean,
            tonic_volatility=exteroceptive_tonic_volatility,
        )
    )

    # set the input data
    extero_network.input_data(input_data=input_data_extero)
    intero_network.input_data(input_data=input_data_intero)

    # response models
    theta_extero = 1.0 - cdf(
        0,
        loc=extero_network.node_trajectories[3]["mean"]
        - extero_network.node_trajectories[1]["mean"],
        scale=jnp.sqrt(2 / extero_network.node_trajectories[2]["precision"]),
    )

    theta_intero = 1.0 - cdf(
        0,
        loc=intero_network.node_trajectories[3]["mean"]
        - intero_network.node_trajectories[1]["mean"],
        scale=jnp.sqrt(
            (1 / intero_network.node_trajectories[0]["precision"])
            + (1 / intero_network.node_trajectories[2]["precision"])
        ),
    )

    thetas = jnp.clip(
        jnp.append(jnp.array(theta_extero), jnp.array(theta_intero)), 1e-12, 1 - 1e-12
    )

    # we have to store the cardiac belief in the array to return it
    cardiac_belief = intero_network.node_trajectories[1]["mean"].mean()

    interoceptive_sensitivity = jnp.log(
        intero_network.node_trajectories[0]["expected_precision"][-1]
    ) - jnp.log(intero_network.node_trajectories[1]["expected_precision"][-1])
    exteroceptive_sensitivity = jnp.log(
        extero_network.node_trajectories[0]["expected_precision"][-1]
    ) - jnp.log(extero_network.node_trajectories[1]["expected_precision"][-1])

    return jnp.append(
        jnp.array(
            [cardiac_belief, interoceptive_sensitivity, exteroceptive_sensitivity]
        ),
        thetas,
    )


def get_cardiac_hgf_op(
    input_data_extero: np.ndarray,
    input_data_intero: np.ndarray,
):
    """Return a PyTensor-compatible cardiac HGF function built with ``wrap_jax``.

    ``pytensor.wrap_jax`` transpiles the JAX computation into a PyTensor graph and
    provides the gradients automatically (via JAX's reverse-mode autodiff), so the
    manual ``Op``/vector-Jacobian-product machinery is no longer required. The
    returned callable accepts PyTensor variables and returns the vector of
    HGF outputs.
    """

    @wrap_jax
    def cardiac_hgf_op(
        interoceptive_precision,
        interoceptive_tonic_volatility,
        interoceptive_mean,
        exteroceptive_precision,
        exteroceptive_tonic_volatility,
        exteroceptive_mean,
    ):
        return cardiac_hgf(
            interoceptive_precision,
            interoceptive_tonic_volatility,
            interoceptive_mean,
            exteroceptive_precision,
            exteroceptive_tonic_volatility,
            exteroceptive_mean,
            input_data_extero=input_data_extero,
            input_data_intero=input_data_intero,
        )

    return cardiac_hgf_op


def sample_cardiac_hgf(
    input_data_extero: np.ndarray,
    input_data_intero: np.ndarray,
    extero_decision: np.ndarray,
    intero_decision: np.ndarray,
    n_cores: int = 1,
):
    """Fit the cardiac hierarchical Gaussian filter to group of participants."""
    cardiac_hgf_op = get_cardiac_hgf_op(
        input_data_extero=input_data_extero,
        input_data_intero=input_data_intero,
    )

    mask = np.append(
        np.ones(input_data_extero.shape[0]), np.zeros(input_data_intero.shape[0])
    ).astype(bool)
    with pm.Model() as model:
        interoceptive_precision = pm.Uniform(
            "interoceptive_precision",
            0.0,
            2.0,
        )

        exteroceptive_precision = pm.Uniform(
            "exteroceptive_precision",
            0.0,
            2.0,
        )

        interoceptive_tonic_volatility = pm.Normal(
            "interoceptive_tonic_volatility",
            0.0,
            30.0,
        )
        exteroceptive_tonic_volatility = pm.Normal(
            "exteroceptive_tonic_volatility",
            0.0,
            30.0,
        )

        exteroceptive_mean = pm.Uniform("exteroceptive_mean", 10.0, 200.0)
        interoceptive_mean = pm.Uniform("interoceptive_mean", 10.0, 200.0)

        hgf_output = pm.Deterministic(
            "hgf_output",
            cardiac_hgf_op(
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
            ),
        )

        thetas = hgf_output[3:]

        _ = pm.Deterministic("cardiac_belief", hgf_output[0])
        _ = pm.Deterministic("interoceptive_sensitivity", hgf_output[1])
        _ = pm.Deterministic("exteroceptive_sensitivity", hgf_output[2])

        _ = pm.Binomial(
            "bin_extero",
            p=thetas[mask],
            n=1,
            observed=extero_decision,
        )
        _ = pm.Binomial(
            "bin_intero",
            p=thetas[~mask],
            n=1,
            observed=intero_decision,
        )

        idata = pm.sample(
            chains=4,
            cores=n_cores,
            draws=1000,
            return_inferencedata=True,
        )

    return idata, model
