import argparse

from mlpy.experiments.run import run
from mlpy.experiments.episodic import EpisodicExperiment
from mlpy.mdp import MDPModelFactory
from mlpy.planners.explorers import ExplorerFactory
from mlpy.planners.discrete import ValueIteration
from mlpy.environments.utils.webots.client import WebotsClient

from penaltykick.environment import PenaltyKickEnvironment
from penaltykick.agent import PenaltyKickAgent


def main(args):
    explorer = None
    if args.explorer_type in ['egreedyexplorer', 'softmaxexplorer']:
        explorer = ExplorerFactory.create(args.explorer_type, args.explorer_params, args.decay)

    a_args = (args.pip, args.pport, args.fsm_config,)
    a_args += ("learningmodule", args.keep_history, None, {"filename": args.file, "append": args.append},
               args.learner,)
    
    if args.learner == 'modelbasedlearner':
        kwargs = {}
        if args.model == 'decisiontreemodel':
            kwargs = {'use_reward_trees': True}
            if args.explorer_type in ['unvisitedbonusexplorer', 'leastvisitedbonusexplorer', 'unknownbonusexplorer']:
                kwargs.update({'explorer_type': args.explorer_type,
                               'rmax': args.rmax})
            if args.explorer_type == 'leastvisitedbonusexplorer' and args.thresh:
                kwargs.update({'thresh': args.thresh})
        else:
            args.ignore_unreachable = False
        model = MDPModelFactory.create(args.model, **kwargs)

        a_args += (ValueIteration(model, explorer, args.gamma, args.ignore_unreachable),)
    else:
        a_args += (explorer,)

    a_kwargs = {'filename': args.savefile, 'progress': args.progress}

    client = None
    if args.client == 'webots':
        client = WebotsClient()

    run(PenaltyKickEnvironment, PenaltyKickAgent, EpisodicExperiment,
        env_args=(args.feature_rep, args.support_leg, client,),
        agent_args=a_args, agent_kwargs=a_kwargs,
        local=args.local)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Learn policy via reinforcement learning.")
    ap.add_argument("--local", action="store_false",
                    help="When true, the experiment is run through a local RLGlue implementation.")
    ap.add_argument("--progress", action="store_true", help="When true, a previously saved trial will be continued.")
    ap.add_argument("--keep_history", action="store_true",
                    help="When true, a history of states and actions performed by agent is saved to file <file>.")
    ap.add_argument("--append", action="store_true",
                    help="When true, observations are appended to the <file>.")
    ap.add_argument("--ignore_unreachable", action="store_true",
                    help="When true, unreachable states are being ignored during planning")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO IP address")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO port")
    ap.add_argument("--client", type=str, choices=["webots", "human"], default="webots", required=False,
                    help="The client the environment connects to.")
    ap.add_argument("--learner", type=str, choices=['qlearner', 'modelbasedlearner'], default="modelbasedlearner",
                    required=False, help="Learning type")
    ap.add_argument("--model", type=str, default="decisiontreemodel", required=False, help="The model type")
    ap.add_argument("--explorer_type", type=str, default="leastvisitedbonusexplorer", required=False,
                    help="The explorer type type to use")
    ap.add_argument("--rmax", type=float, default=20.0, required=False, help="The maximum possible reward")
    ap.add_argument("--thresh", type=float, required=False, help="Model explorer threshold")
    # ap.add_argument("--explorer_params", '--list', nargs='+', required=False,
    #                 help="List of explorer parameters")
    ap.add_argument("--explorer_params", default=0.5, type=float, required=False, help="Epsilon or tau")
    ap.add_argument("--decay", type=float, default=1.0, required=False, help="The decay value for exploration")
    ap.add_argument("--gamma", type=float, default=0.99, required=False, help="The discount factor")
    ap.add_argument("--support_leg", type=str, default="right", required=False,
                    help="The supporting leg. The information must match with the supporting leg in "
                         "the FSM configuration")
    ap.add_argument("--feature_rep", type=str, choices=["rl", "irl"], default="rl", required=False,
                    help="The feature representation to use for the experiment.")
    ap.add_argument("--fsm_config", type=str, required=True,
                    help="Name of the file containing the finite state machine configuration.")
    ap.add_argument("--savefile", type=str, required=False,
                    help="The name of the file containing the learner's saved trial.")
    ap.add_argument("--file", type=str, required=False, help="Name of the file to store the recorded data to.")

    main(ap.parse_args())
