import jax.numpy as jnp
from typing import Union

ArrayLike = Union[float, jnp.ndarray]


def _to_array(val: ArrayLike) -> jnp.ndarray:
    """Convert input to jnp.ndarray if it's not already.

    Args:
        val: A float or jnp.ndarray.

    Returns:
        A jnp.ndarray.
    """
    return val if isinstance(val, jnp.ndarray) else jnp.array(val)


def _match_type(val: ArrayLike, result: jnp.ndarray) -> ArrayLike:
    """Return result in the same type as input.

    Args:
        val: Original input (float or jnp.ndarray).
        result: Computed result as jnp.ndarray.

    Returns:
        Result as float if input was float, or as jnp.ndarray if input was jnp.ndarray.
    """
    return float(result) if isinstance(val, float) else result


def kph2mps(val: ArrayLike) -> ArrayLike:
    """Convert speed from kilometers per hour (kph) to meters per second (mps).

    Args:
        val: Speed in kilometers per hour (float or jnp.ndarray).

    Returns:
        Speed in meters per second, matching the input type.
    """
    result = _to_array(val) / 3.6
    return _match_type(val, result)


def mps2kph(val: ArrayLike) -> ArrayLike:
    """Convert speed from meters per second (mps) to kilometers per hour (kph).

    Args:
        val: Speed in meters per second (float or jnp.ndarray).

    Returns:
        Speed in kilometers per hour, matching the input type.
    """
    result = _to_array(val) * 3.6
    return _match_type(val, result)


def deg2rad(val: ArrayLike) -> ArrayLike:
    """Convert angle(s) from degrees to radians.

    Args:
        val: Angle(s) in degrees (float or jnp.ndarray).

    Returns:
        Angle(s) in radians, matching the input type.
    """
    result = _to_array(val) * jnp.pi / 180.0
    return _match_type(val, result)


def rad2deg(val: ArrayLike) -> ArrayLike:
    """Convert angle(s) from radians to degrees.

    Args:
        val: Angle(s) in radians (float or jnp.ndarray).

    Returns:
        Angle(s) in degrees, matching the input type.
    """
    result = _to_array(val) * 180.0 / jnp.pi
    return _match_type(val, result)


def mod2pi(angle: ArrayLike) -> ArrayLike:
    """Normalize angle(s) to the range [-π, π].

    Args:
        angle: Angle(s) in radians (float or jnp.ndarray).

    Returns:
        Normalized angle(s) in the range [-π, π], matching the input type.
    """
    result = ( _to_array(angle) + jnp.pi ) % (2 * jnp.pi) - jnp.pi
    return _match_type(angle, result)
