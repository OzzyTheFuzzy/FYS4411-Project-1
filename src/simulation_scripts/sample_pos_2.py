import sys
from pathlib import Path

import jax

project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
data_dir = project_root / "positions_energy_data"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "simulation_scripts"))

from qs.functions import vmc_and_exact_energy as vmc_and_exact_energy
import config_sample_pos_2 as spherical_config
import config_sample_pos_2_anisotropic as anisotropic_config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def run_case(config, dim, nparticles, a):


    output_file = data_dir / f"r_all_N{nparticles}_d{dim}_beta{config.beta}_a{a}.dat"
    if output_file.exists():
        print(f"Skipping existing file: {output_file.name}")
        return None, None
    config.dim=dim
    config.nparticles=nparticles
    config.a=a
    system = vmc_and_exact_energy.find_energy_vmc(dim, nparticles, config, config.scale)
    system._make_initial_state()
    results = system.sample(
        config.nsamples,
        config.final_burn_in,
        nchains=config.nchains,
        seed=24,
        num=config.num,
        write_to_file=config.write_to_file,
        name_of_file=config.name_of_file,
    )
    return system, results


def main():
    sweeps = [
        (
            spherical_config,
            [
                (1, 1,  0.0),
                (3, 1,  0.0),
                (1, 2,  0.0),
                (3, 2,  0.0),
                (3, 10, 0.0),
            ],
        ),
        (
            anisotropic_config,
            [
                (3, 2,  0.0),
                (3, 10, 0.0),
            ],
        ),
    ]

    for config, cases in sweeps:
        trap = "anisotropic" if config.beta is not None else "spherical"
        for dim, nparticles, a in cases:
            print(f"Running {trap} case: N={nparticles}, d={dim}, beta={config.beta}, a={a}")
            run_case(config, dim, nparticles, a)


if __name__ == "__main__":
    main()