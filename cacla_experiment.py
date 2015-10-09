import argparse
import numpy as np

from mlpy.experiments.run import run
from mlpy.experiments.episodic import EpisodicExperiment
from mlpy.environments.cartpole import CartPole
from mlpy.agents.modelbased import ModelBasedAgent


def main(args):

    if args.single_pole:
        if args.friction:
            # single pole with friction
            run(CartPole, ModelBasedAgent, EpisodicExperiment,
                env_args=('custom', [.5], [.1], 10),
                env_kwargs={'random_start': False, 'cart_loc': [-2.4, 2.4], 'cart_vel': [-np.inf, np.inf],
                            'pole_angle': [-36, 36], 'pole_vel': [-np.inf, np.inf], 'mu_c': 0.0005, 'mu_p': 0.000002,
                            'discount_factor': 0.99},
                agent_args=('learningmodule', False, None, None, None, 'cacla', 0, 40),
                agent_kwargs={'alpha': 0.001, 'beta': 0.001},
                exp_args=(1, 1000, 1000),
                exp_kwargs={'filename': args.file},
                local=True)
        else:
            run(CartPole, ModelBasedAgent, EpisodicExperiment,
                env_args=('custom', [.5], [.1], 10),
                env_kwargs={'random_start': False, 'cart_loc': [-2.4, 2.4], 'cart_vel': [-np.inf, np.inf],
                            'pole_angle': [-36, 36], 'pole_vel': [-np.inf, np.inf], 'mu_c': 0.0005, 'mu_p': 0.000002,
                            'discount_factor': 0.99, 'include_friction': False},
                agent_args=('learningmodule', False, None, None, None, 'cacla', 0, 40),
                agent_kwargs={'alpha': 0.001, 'beta': 0.001},
                exp_args=(1, 500, 1000),
                exp_kwargs={'filename': args.file},
                local=True)
    else:
        # double pole with friction
        run(CartPole, ModelBasedAgent, EpisodicExperiment,
            env_args=('custom', [1., .1], [.1, .01], 10),
            env_kwargs={'discrete_action': False, 'random_start': True, 'cart_loc': [-2.4, 2.4],
                        'cart_vel': [-np.inf, np.inf], 'pole_angle': [-36., 36.], 'pole_vel': [-np.inf, np.inf],
                        'mu_c': 0.0005, 'mu_p': 0.000002, 'discount_factor': 0.99},
            agent_args=('learningmodule', False, None, None, None, 'cacla', 0, 40),
            agent_kwargs={'alpha': 0.001, 'beta': 0.001},
            exp_args=(1, 200000, 1000),
            exp_kwargs={'filename': args.file},
            local=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Experiment: Learn policy via reinforcement learning.")
    ap.add_argument("--single_pole", action="store_true",
                    help="When set, the environment is cart pole with a single pole.")
    ap.add_argument("--friction", action="store_true", help="When set, the dynamics include fiction.")
    ap.add_argument("--file", type=str, required=False, help="Name of the file to store the recorded data to.")

    main(ap.parse_args())
