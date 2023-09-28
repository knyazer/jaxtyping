import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest
from beartype import *  # noqa: F403

from jaxtyping import Float32

from .helpers import ParamError


def g(x: Float32[jnp.ndarray, " b"]):
    pass


g(jnp.array([1.0]))
with pytest.raises(ParamError):
    g(jnp.array(1))


class M(eqx.Module):
    foo: int
    bar: Float32[jnp.ndarray, " a"]


M(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    M(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    M(1, jnp.array(1.0))


@dataclasses.dataclass
class D:
    foo: int
    bar: Float32[jnp.ndarray, " a"]


D(1, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1.0, jnp.array([1.0]))
with pytest.raises(ParamError):
    D(1, jnp.array(1.0))
