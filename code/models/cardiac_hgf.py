# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.model import Network
from jax import jit, Array, vjp
from jax.scipy.stats.norm import cdf
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jax.tree_util import Partial
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import pymc as pm


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
        Network(update_type="unbounded")
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
        Network(update_type="unbounded")
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

    return jnp.clip(
        jnp.append(jnp.array(theta_extero), jnp.array(theta_intero)), 1e-12, 1 - 1e-12
    )


def get_cardiac_hgf_op(
    input_data_extero: np.ndarray,
    input_data_intero: np.ndarray,
):
    """Get the cardiac HGF custom operation."""
    # create the partial function for the group level cardiac HGF
    partial_cardiac_hgf = Partial(
        cardiac_hgf,
        input_data_extero=input_data_extero,
        input_data_intero=input_data_intero,
    )

    jitted_cardiac_hgf = jit(partial_cardiac_hgf)

    def vjp_custom_op_jax(
        interoceptive_precision: ArrayLike,
        interoceptive_tonic_volatility: ArrayLike,
        interoceptive_mean: ArrayLike,
        exteroceptive_precision: ArrayLike,
        exteroceptive_tonic_volatility: ArrayLike,
        exteroceptive_mean: ArrayLike,
        cotangent: ArrayLike,
    ):
        """Get the custom vector Jacobian product for the group level cardiac HGF."""
        _, vjp_fn = vjp(
            partial_cardiac_hgf,
            interoceptive_precision,
            interoceptive_tonic_volatility,
            interoceptive_mean,
            exteroceptive_precision,
            exteroceptive_tonic_volatility,
            exteroceptive_mean,
        )
        return vjp_fn(cotangent)

    jitted_vjp_group_level_cardiac_hgf = jit(vjp_custom_op_jax)

    # The CustomOp needs `make_node`, `perform` and `grad`.
    class CustomOp(Op):
        """Custom Op for the cardiac HGF."""

        def make_node(
            self,
            interoceptive_precision,
            interoceptive_tonic_volatility,
            interoceptive_mean,
            exteroceptive_precision,
            exteroceptive_tonic_volatility,
            exteroceptive_mean,
        ):
            """Create a node for the cardiac HGF."""
            # Create a PyTensor node specifying the number and type of inputs / outputs

            # We convert the input into a PyTensor tensor variable
            inputs = [
                pt.as_tensor_variable(interoceptive_precision),
                pt.as_tensor_variable(interoceptive_tonic_volatility),
                pt.as_tensor_variable(interoceptive_mean),
                pt.as_tensor_variable(exteroceptive_precision),
                pt.as_tensor_variable(exteroceptive_tonic_volatility),
                pt.as_tensor_variable(exteroceptive_mean),
            ]
            outputs = [pt.vector(dtype="float64")]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            """Perform the operation defined by the Op."""
            # Evaluate the Op result for a specific numerical input

            (
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
            ) = inputs
            result = jitted_cardiac_hgf(
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
            )
            # The results should be assigned inplace to the nested list
            # of outputs provided by PyTensor. If you have multiple
            # outputs and results, you should assign each at outputs[i][0]
            outputs[0][0] = np.asarray(result, dtype="float64")

        def grad(self, inputs, output_gradients):
            """Create a PyTensor expression of the gradient."""
            # Create a PyTensor expression of the gradient
            (cotangent,) = output_gradients
            # We reference the VJP Op created below, which encapsulates
            # the gradient operation
            return vjp_custom_op(*inputs, cotangent)

    class VJPCustomOp(Op):
        """Vector-Jacobian product for the cardiac HGF."""

        def make_node(
            self,
            interoceptive_precision,
            interoceptive_tonic_volatility,
            interoceptive_mean,
            exteroceptive_precision,
            exteroceptive_tonic_volatility,
            exteroceptive_mean,
            cotangent,
        ):
            """Create a node for the vector-Jacobian product of the cardiac HGF."""
            # Make sure the two inputs are tensor variables
            inputs = [
                pt.as_tensor_variable(interoceptive_precision),
                pt.as_tensor_variable(interoceptive_tonic_volatility),
                pt.as_tensor_variable(interoceptive_mean),
                pt.as_tensor_variable(exteroceptive_precision),
                pt.as_tensor_variable(exteroceptive_tonic_volatility),
                pt.as_tensor_variable(exteroceptive_mean),
                pt.as_tensor_variable(cotangent),
            ]
            outputs = [pt.scalar(dtype="float64") for _ in range(6)]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            """Evaluate the Op result for a specific numerical input."""
            (
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
                cotangent,
            ) = inputs
            result = jitted_vjp_group_level_cardiac_hgf(
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
                cotangent,
            )
            for i in range(6):
                outputs[i][0] = np.asarray(result[i], dtype="float64")

    # Instantiate the Ops
    cardiac_hgf_op = CustomOp()
    vjp_custom_op = VJPCustomOp()

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

        thetas = pm.Deterministic(
            "thetas",
            cardiac_hgf_op(
                interoceptive_precision,
                interoceptive_tonic_volatility,
                interoceptive_mean,
                exteroceptive_precision,
                exteroceptive_tonic_volatility,
                exteroceptive_mean,
            ),
        )

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
