import torch

from pathlib import Path

project_root = Path(__file__).resolve().parents[3]


def accumulate_column_density(r, n_bins=25, r_max=None):
    """
    Compute column density histogram from saved Monte Carlo positions.

    r : (N, dim) Tensor of particle positions.
    n_bins : int Number of radial bins in the xy-plane.
    r_max : float or None Maximum transverse radius. If None, use 99.5% quantile.
    """

    r = r.to(torch.float64)
    N, dim = r.shape

    if dim < 2:
        raise ValueError("Column density requires at least 2 dimensions")

    # transverse radius (distance from z-axis)
    x = r[:, 0]
    y = r[:, 1]
    r_perp = torch.sqrt(x**2 + y**2)

    if r_max is None:
        r_max = torch.quantile(r_perp, 0.995).item()

    # histogram bins
    edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=torch.float64)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # bin counts
    bin_idx = torch.bucketize(r_perp, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    counts = torch.bincount(bin_idx[valid], minlength=n_bins).to(torch.float64)

    # annulus areas in xy-plane
    annulus_areas = torch.pi * (edges[1:]**2 - edges[:-1]**2)

    return r_centers, counts, annulus_areas


def compute_column_density(counts_accumulated, annulus_areas, N, nsamples):
    """
    Compute column density after accumulating over all MC cycles.

    Returns density normalized so that ∫ n_c(r_perp) dA = 1
    """
    n_c = counts_accumulated / (annulus_areas * N * nsamples)
    return n_c

def plot_column_density(r_centers, n_c, config):
    """
    Plot the column density.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(r_centers, n_c, marker="o")
    plt.xlabel("r_perp")
    plt.ylabel("n_c(r_perp)")
    plt.xlim(0, 3.0)        # adjust if needed
    plt.ylim(-0.01, 0.35)        
    plt.title(f"Column density for N={config.nparticles} and a={config.a}")
    plt.grid(True)
    plt.savefig(project_root / "figures" / f"column_density_N{config.nparticles}_samples{config.nsamples}.pdf", dpi=300)
    plt.show()

