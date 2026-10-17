"""Unit labels for type hints. To Python they are just float and array;
they tell the reader what a number means, e.g. `yaw: Radians`."""

from typing import TypeAlias

import numpy as np

Metres: TypeAlias = float
MetresPerSecond: TypeAlias = float
Radians: TypeAlias = float
Seconds: TypeAlias = float
PerMetre: TypeAlias = float     # curvature: 1 / turn radius
Array: TypeAlias = np.ndarray   # numpy in the driver, jax.numpy in training
