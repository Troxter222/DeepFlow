# """
# Script to generate a synthetic dataset for the unsteady (decaying) two‑dimensional
# Taylor‑Green vortex.  The Taylor‑Green vortex is a canonical test problem for the
# incompressible Navier–Stokes equations.  In two dimensions, its exact solution
# for the velocity components and pressure is given on the domain \([-\pi,\pi]^2\) by

#   * `u(x, y, t) = sin(x) * cos(y) * exp(-2 * nu * t)`
#   * `v(x, y, t) = -cos(x) * sin(y) * exp(-2 * nu * t)`
#   * `p(x, y, t) = 0.25 * (cos(2*x) + cos(2*y)) * exp(-4 * nu * t)`

# These expressions match the classical Taylor–Green vortex solution, where the
# velocity field is given by \(u=\sin x\cos y\,F(t)\) and \(v=-\cos x\sin y\,F(t)\) with
# \(F(t) = \exp(-2\nu t)\)【323460344680131†L170-L185】.  The pressure field can be derived by
# substituting the velocity into the momentum equations, yielding
# \(p=(1/4)(\cos 2x + \cos 2y)F(t)^2\)【323460344680131†L191-L195】.

# This script samples points uniformly in space and time and computes the
# corresponding velocity and pressure values.  It writes the resulting data to
# a CSV file.  To avoid excessive memory usage when generating millions of
# points, the data is written in smaller chunks.

# Usage:
#     python generate_taylor_green_dataset.py --n_samples 5000000 --outfile dataset.csv

# Command-line arguments:
#     --n_samples N     Number of sample points to generate (default: 5_000_000).
#     --nu NU           Kinematic viscosity ν used in the exact solution (default: 0.1).
#     --x_min A         Minimum x coordinate (default: -π).
#     --x_max B         Maximum x coordinate (default: π).
#     --y_min C         Minimum y coordinate (default: -π).
#     --y_max D         Maximum y coordinate (default: π).
#     --t_min E         Minimum time value (default: 0.0).
#     --t_max F         Maximum time value (default: 2.0).
#     --chunk_size K    Number of points to process per chunk (default: 500_000).
#     --seed S          Optional seed for the RNG to make results reproducible.
#     --outfile PATH    Path to the output CSV file (default: taylor_green_dataset.csv).

# The resulting CSV has six columns: x, y, t, u, v, p.
# """

# import argparse
# import csv
# import math
# from typing import Tuple

# import numpy as np


# def generate_chunk(
#     rng: np.random.Generator,
#     n: int,
#     x_range: Tuple[float, float],
#     y_range: Tuple[float, float],
#     t_range: Tuple[float, float],
#     nu: float,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Generate one chunk of the Taylor–Green vortex dataset.

#     Parameters
#     ----------
#     rng : np.random.Generator
#         Random number generator instance.
#     n : int
#         Number of points in the chunk.
#     x_range, y_range : Tuple[float, float]
#         The lower and upper bounds for the x and y coordinates.
#     t_range : Tuple[float, float]
#         The lower and upper bounds for the time coordinate.
#     nu : float
#         Kinematic viscosity.

#     Returns
#     -------
#     x, y, t, u, v, p : np.ndarray
#         Arrays of shape `(n,)` containing the coordinates and corresponding
#         velocity and pressure values.
#     """
#     # Uniformly sample spatial and temporal coordinates
#     x = rng.uniform(x_range[0], x_range[1], size=n)
#     y = rng.uniform(y_range[0], y_range[1], size=n)
#     t = rng.uniform(t_range[0], t_range[1], size=n)
#     # Compute the decaying factor F(t) = exp(-2 * nu * t) and F^2(t) = exp(-4 * nu * t)
#     f_t = np.exp(-2.0 * nu * t)
#     f2_t = f_t * f_t  # more efficient than np.exp(-4*nu*t)
#     # Velocity components
#     u = np.sin(x) * np.cos(y) * f_t
#     v = -np.cos(x) * np.sin(y) * f_t
#     # Pressure field
#     p = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * f2_t
#     return x, y, t, u, v, p


# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Generate a synthetic dataset for the unsteady Taylor–Green vortex "
#             "decaying solution. See the module docstring for details."
#         )
#     )
#     parser.add_argument("--n_samples", type=int, default=5_000_000, help="Total number of points to generate.")
#     parser.add_argument("--nu", type=float, default=0.1, help="Kinematic viscosity ν.")
#     parser.add_argument("--x_min", type=float, default=-math.pi, help="Minimum x coordinate.")
#     parser.add_argument("--x_max", type=float, default=math.pi, help="Maximum x coordinate.")
#     parser.add_argument("--y_min", type=float, default=-math.pi, help="Minimum y coordinate.")
#     parser.add_argument("--y_max", type=float, default=math.pi, help="Maximum y coordinate.")
#     parser.add_argument("--t_min", type=float, default=0.0, help="Minimum time value.")
#     parser.add_argument("--t_max", type=float, default=2.0, help="Maximum time value.")
#     parser.add_argument(
#         "--chunk_size",
#         type=int,
#         default=500_000,
#         help="Number of samples to generate and write per chunk. Must be positive.",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=None,
#         help="Optional seed for the random number generator to ensure reproducibility."
#     )
#     parser.add_argument(
#         "--outfile",
#         type=str,
#         default="taylor_green_dataset.csv",
#         help="Output CSV filename."
#     )

#     args = parser.parse_args()

#     # Validate chunk size
#     if args.chunk_size <= 0:
#         raise ValueError("chunk_size must be positive.")

#     rng = np.random.default_rng(args.seed)
#     total_samples = args.n_samples
#     chunk_size = args.chunk_size

#     x_range = (args.x_min, args.x_max)
#     y_range = (args.y_min, args.y_max)
#     t_range = (args.t_min, args.t_max)

#     # Open the CSV file and write header
#     with open(args.outfile, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         # Header row
#         writer.writerow(["x", "y", "t", "u", "v", "p"])
#         n_written = 0
#         while n_written < total_samples:
#             n_to_generate = min(chunk_size, total_samples - n_written)
#             x, y, t, u, v, p = generate_chunk(rng, n_to_generate, x_range, y_range, t_range, args.nu)
#             # Using zip is memory efficient and works well for CSV writing
#             writer.writerows(zip(x, y, t, u, v, p))
#             n_written += n_to_generate
#             # Optionally print progress to the console
#             print(f"Generated and wrote {n_written} / {total_samples} points", end="\r")

#     print(f"\nDataset generation complete. File saved to {args.outfile}")


# if __name__ == "__main__":
#     main()


import numpy as np
import os

def generate_taylor_green(n_points=1000000, nu=0.1):
    print("Generating Taylor-Green Vortex (Transient)...")
    print(f"Target: {n_points} points, Viscosity (nu) = {nu}")
    
    x = np.random.uniform(-np.pi, np.pi, n_points).astype(np.float32)
    y = np.random.uniform(-np.pi, np.pi, n_points).astype(np.float32)
    t = np.random.uniform(0, 2.0, n_points).astype(np.float32)
    
    # F(t) = e^(-2 * nu * t)
    F_t = np.exp(-2 * nu * t)
    
    u = np.sin(x) * np.cos(y) * F_t
    v = -np.cos(x) * np.sin(y) * F_t
    
    p = 0.25 * (np.cos(2*x) + np.cos(2*y)) * (F_t**2)
    
    data = np.column_stack((t, x, y, u, v, p))
    
    # 4. Save
    os.makedirs("data", exist_ok=True)
    save_path = "data/taylor_green.npy"
    
    np.save(save_path, data)
    print(f"✅ Success! Saved to {save_path}")
    print(f"📊 File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    generate_taylor_green(1000000, nu=0.1)