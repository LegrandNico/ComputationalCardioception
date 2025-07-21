# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.model import Network
from jax.tree_util import Partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
import pytensor.tensor as pt
from jax import jit, grad
from pytensor.graph.op import Op
from pyhgf.math import binary_surprise
from pytensor.graph.basic import Apply
from jax.typing import ArrayLike
from typing import Callable


def response_function(
    hgf: Network,
    response_function_inputs: tuple,
    response_function_parameters: tuple,
    pointwise: bool = False,
):
    """Response function."""
    intero_decision, intero_tone_2, extero_decision, extero_tone_1, extero_tone_2 = (
        response_function_inputs
    )
    auditory_precision = response_function_parameters[0]

    # exteroception --------------------------------------------------------------------
    threshold = extero_tone_2 - extero_tone_1
    slope = jnp.sqrt(2 / auditory_precision)
    extero_surprise = binary_surprise(
        x=extero_decision, expected_mean=1.0 - norm.cdf(0, loc=threshold, scale=slope)
    )

    # interoception --------------------------------------------------------------------
    threshold = intero_tone_2 - hgf.node_trajectories[1]["mean"]
    slope = jnp.sqrt(
        (1 / auditory_precision) + (1 / hgf.node_trajectories[0]["precision"])
    )
    intero_surprise = binary_surprise(
        x=intero_decision, expected_mean=1.0 - norm.cdf(0, loc=threshold, scale=slope)
    )
    if pointwise:
        return jnp.concat([extero_surprise, intero_surprise])
    else:
        return jnp.sum(jnp.concat([extero_surprise, intero_surprise]))


def cardioception_hgf_logp(
    tonic_volatility: float,
    cardiac_precision: float,
    cardiac_mean: float,
    auditory_precision: float,
    cardioception_hgf: Network,
    input_data: np.ndarray,
    observed: ArrayLike,
    response_function_inputs: tuple,
    response_function: Callable = response_function,
) -> float:
    """Log probability of the cardioception HGF model."""
    cardioception_hgf.attributes[0]["precision"] = cardiac_precision
    cardioception_hgf.attributes[0]["expected_precision"] = cardiac_precision
    cardioception_hgf.attributes[1]["tonic_volatility"] = tonic_volatility
    cardioception_hgf.attributes[1]["mean"] = cardiac_mean

    return -cardioception_hgf.input_data(
        input_data=input_data,
        observed=observed,
    ).surprise(
        response_function=response_function,
        response_function_inputs=response_function_inputs,
        response_function_parameters=(auditory_precision,),
    )


def get_cardioception_hgf(
    input_data: ArrayLike,
    observed: ArrayLike,
    response_function_inputs: ArrayLike,
    response_function: Callable = response_function,
    pointwise: bool = False,
):
    """Get the log probability function as PyTensor Op."""
    # create the probabilistic network
    cardioception_hgf = (
        Network(update_type="unbounded")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
    )

    response_function = Partial(response_function, pointwise=pointwise)

    logp_fn = Partial(
        cardioception_hgf_logp,
        cardioception_hgf=cardioception_hgf,
        input_data=input_data,
        observed=observed,
        response_function=response_function,
        response_function_inputs=response_function_inputs,
    )

    jitted_custom_op_jax = jit(logp_fn)
    if not pointwise:
        grad_logp_fn = jit(grad(logp_fn, argnums=[0, 1, 2, 3]))

    # The CustomOp needs `make_node`, `perform` and `grad`.
    class CustomOp(Op):
        def make_node(
            self,
            tonic_volatility: float,
            cardiac_precision: float,
            cardiac_mean: float,
            auditory_precision: float,
        ):
            # We convert the input into a PyTensor tensor variable
            inputs = [
                pt.as_tensor_variable(tonic_volatility),
                pt.as_tensor_variable(cardiac_precision),
                pt.as_tensor_variable(cardiac_mean),
                pt.as_tensor_variable(auditory_precision),
            ]
            # Output has the same type and shape as `x`
            if pointwise:
                outputs = [pt.vector(dtype=float)]
            else:
                outputs = [pt.scalar(dtype=float)]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            # Evaluate the Op result for a specific numerical input

            # The inputs are always wrapped in a list
            result = jitted_custom_op_jax(*inputs)
            # The results should be assigned in place to the nested list
            # of outputs provided by PyTensor. If you have multiple
            # outputs and results, you should assign each at outputs[i][0]
            outputs[0][0] = np.asarray(result, dtype="float64")

        def grad(self, inputs, output_gradients):
            # Create a PyTensor expression of the gradient
            (
                grad_tonic_volatility,
                grad_cardiac_precision,
                grad_cardiac_mean,
                grad_auditory_precision,
            ) = grad_custom_op(*inputs)

            output_gradient = output_gradients[0]
            # We reference the VJP Op created below, which encapsulates
            # the gradient operation
            return [
                output_gradient * grad_tonic_volatility,
                output_gradient * grad_cardiac_precision,
                output_gradient * grad_cardiac_mean,
                output_gradient * grad_auditory_precision,
            ]

    class GradCustomOp(Op):
        def make_node(
            self,
            tonic_volatility: float,
            cardiac_precision: float,
            cardiac_mean: float,
            auditory_precision: float,
        ):
            # Make sure the two inputs are tensor variables
            inputs = [
                pt.as_tensor_variable(tonic_volatility),
                pt.as_tensor_variable(cardiac_precision),
                pt.as_tensor_variable(cardiac_mean),
                pt.as_tensor_variable(auditory_precision),
            ]
            # Output has the shape type and shape as the first input
            outputs = [inp.type() for inp in inputs]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            (
                grad_tonic_volatility,
                grad_cardiac_precision,
                grad_cardiac_mean,
                grad_auditory_precision,
            ) = grad_logp_fn(*inputs)

            outputs[0][0] = np.asarray(
                grad_tonic_volatility, dtype=node.outputs[0].dtype
            )
            outputs[1][0] = np.asarray(
                grad_cardiac_precision, dtype=node.outputs[1].dtype
            )
            outputs[2][0] = np.asarray(grad_cardiac_mean, dtype=node.outputs[2].dtype)
            outputs[3][0] = np.asarray(
                grad_auditory_precision, dtype=node.outputs[3].dtype
            )

    # Instantiate the Ops
    custom_op = CustomOp()
    grad_custom_op = GradCustomOp()

    return custom_op
