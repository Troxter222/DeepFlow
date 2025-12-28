"""
Script to generate a synthetic dataset for the steady Kovasznay flow.

This script samples a specified number of random points uniformly in a
rectangular domain and evaluates the exact analytical solution of the
Kovasznay flow at each point.  The resulting dataset is saved as a NumPy
binary file (``.npy``) with columns [t, x, y, u, v].  The time column is
filled with zeros since the flow is steady, but its presence makes the
dataset compatible with code that expects a temporal dimension.

The velocity field (u, v) is computed using the following exact solution
for the Kovasznay flow, where ``Re`` is the Reynolds number:

    u(x, y) = 1 – e^(λ x) cos(2π y)
    v(x, y) = (λ / (2π)) e^(λ x) sin(2π y)

The parameter λ is the negative root of the quadratic equation
λ² – Re λ – 4π² = 0, which gives λ = 0.5 Re – sqrt(0.25 Re² + 4π²).  This
definition aligns with the exact solutions used to validate numerical
Navier–Stokes solvers【148347336539100†L126-L136】【289197091849491†L44-L58】.

The default domain and Reynolds number correspond to the specifications
requested in the task: Re = 20, x ∈ [–0.5, 1.0], y ∈ [–0.5, 1.5], and
1 000 000 sample points.
"""

import numpy as np


def generate_kovasznay_flow_data(
    n_points: int = 1_000_000,
    Re: float = 20.0,
    x_min: float = -0.5,
    x_max: float = 1.0,
    y_min: float = -0.5,
    y_max: float = 1.5,
) -> np.ndarray:
    """Generate synthetic Kovasznay flow data.

    Parameters
    ----------
    n_points : int
        Number of random sample points to generate.
    Re : float
        Reynolds number used in the analytical solution.
    x_min, x_max : float
        Minimum and maximum bounds of the domain in the x‑direction.
    y_min, y_max : float
        Minimum and maximum bounds of the domain in the y‑direction.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_points, 5) with columns [t, x, y, u, v],
        stored in ``float32`` for memory efficiency.

    Notes
    -----
    The function uses a pseudorandom number generator (PCG64) for uniform
    sampling.  No seed is set by default to allow different runs to
    produce different datasets.  If reproducibility is required, set the
    seed on the returned generator before calling this function.
    """
    # Compute the negative root λ of the quadratic equation λ² – Re λ – 4π² = 0
    # λ = 0.5*Re – sqrt(0.25*Re*Re + 4*π²).  This value ensures decay of
    # perturbations downstream and matches the classic definition of the
    # Kovasznay flow parameter【148347336539100†L126-L136】.
    lam: float = 0.5 * Re - np.sqrt(0.25 * Re * Re + 4.0 * np.pi * np.pi)

    # Instantiate a random number generator for uniform sampling in the domain.
    rng = np.random.default_rng()

    # Generate random coordinates uniformly within the specified domain.  Use
    # ``float32`` dtype to reduce memory footprint.  Note that the uniform
    # sampler returns ``float64`` by default, so we cast to ``float32``.
    x = rng.uniform(low=x_min, high=x_max, size=n_points).astype(np.float32)
    y = rng.uniform(low=y_min, high=y_max, size=n_points).astype(np.float32)

    # Precompute the exponential term e^(λ x) as float32 to avoid repeated
    # evaluations and to save memory.  Multiplying λ (float64) by x (float32)
    # promotes the result to float64; the exponent is then cast back to
    # float32.
    exp_lx = np.exp(lam * x).astype(np.float32)

    # Compute the u and v velocity components according to the analytical
    # solution.  Each operation is cast to float32 for consistency.
    # u(x, y) = 1 – exp(lam * x) * cos(2π y)
    u = (1.0 - exp_lx * np.cos(2.0 * np.pi * y)).astype(np.float32)

    # v(x, y) = (lam / (2π)) * exp(lam * x) * sin(2π y)
    # Compute the scalar coefficient lam/(2π) once to avoid redundant work.
    lam_over_2pi: float = lam / (2.0 * np.pi)
    v = (lam_over_2pi * exp_lx * np.sin(2.0 * np.pi * y)).astype(np.float32)

    # Create the time column of zeros; dtype float32 to match other columns.
    t = np.zeros(n_points, dtype=np.float32)

    # Stack columns into a single array with the required order: [t, x, y, u, v].
    data = np.column_stack((t, x, y, u, v))
    return data


def main() -> None:
    """Generate data and save it to 'flow_data.npy'."""
    data = generate_kovasznay_flow_data()
    # Save the array using numpy.save.  This writes a binary file that can be
    # loaded with numpy.load("flow_data.npy") in a later session.
    np.save("flow_data.npy", data)


if __name__ == "__main__":
    main()