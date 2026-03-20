import torch

def accumulate_onebody_density(r, n_bins=80, r_max=None):
    """
    Compute radial one-body density from saved Monte Carlo positions.

    r : (N, dim) Tensor of particle positions.
    n_bins : int Number of radial bins.
    r_max : float or None Maximum radius included in histogram. If None, use 99.5% quantile.
    """

    r = r.to(torch.float64)
    N, dim = r.shape[0], r.shape[1]

    radii = torch.linalg.norm(r, dim=1).reshape(-1)

    if r_max is None:
        r_max = torch.quantile(radii, 0.995).item()

    # create edges for the histogram and compute bin centers
    edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=torch.float64)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = torch.bucketize(radii, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    counts = torch.bincount(bin_idx[valid], minlength=n_bins).to(torch.float64)

    # find onebody densities for different number of dimension
    if dim == 1:
        shell_volumes = edges[1:] - edges[:-1]
    elif dim == 2:
        shell_volumes = torch.pi * (edges[1:]**2 - edges[:-1]**2)
    elif dim == 3:
        shell_volumes = (4.0 * torch.pi / 3.0) * (edges[1:]**3 - edges[:-1]**3)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    # Return raw counts and shell volumes — normalization happens after all cycles
    return r_centers, counts, shell_volumes


def compute_onebody_density(counts_accumulated, shell_volumes, N, nsamples):
    """
    Compute the total one-body density after accumulating counts over all MC cycles.

    counts_accumulated : (n_bins,) Tensor — sum of counts over all cycles
    shell_volumes      : (n_bins,) Tensor — bin shell volumes (constant across cycles)
    N                  : int — number of particles
    nsamples           : int — number of MC cycles accumulated
    
    Returns rho normalized so that integral rho(r) dV = N (particle density).
    """
    rho = counts_accumulated / (shell_volumes * N * nsamples)
    return rho

def plot_onebody_density(r_centers, rho, config):
    """
    Plot the one-body density.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(r_centers.numpy(), rho.numpy(), marker="o")
    plt.xlabel("r")
    plt.ylabel("rho(r)")
    plt.title(f"One-body density for {config.mcmc_alg} algorithm, N={config.nparticles}, dim={config.dim}")
    plt.grid(True)
    plt.show()