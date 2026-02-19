

def exact_energy(N, omega_ho, D, omega_z=None):
    """ Calculate the exact emergy for N non-interacting particles in a harmonic oscillator potential
    Args:
        N (int): number of particles
        omega_ho (float): frequency of the harmonic oscillator potential.
        omega_z (float, optional): frequency of the harmonic oscillator potential in the z direction.
        D (int, optional): dimension of the system.
    """

    if omega_z is None:
        return N * omega_ho * D * 0.5
    else:
        return 0.5 * N * (omega_ho * 2 + omega_z)
