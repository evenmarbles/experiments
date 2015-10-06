import sys
import argparse

import numpy as np

from mlpy.auxiliary.io import load_from_file
from mlpy.experiments.run import run
from mlpy.experiments.episodic import EpisodicExperiment
from mlpy.mdp import MDPModelFactory
from mlpy.planners.explorers import ExplorerFactory
from mlpy.planners.discrete import ValueIteration
from mlpy.environments.utils.webots.client import WebotsClient

from mlpy.learners import ApprenticeshipLearner

from penaltykick.environment import PenaltyKickEnvironment
from penaltykick.agent import PenaltyKickAgent


def main(args):
    try:
        data = load_from_file(args.demofile)
        demo = data["state"]
    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    kwargs = {}
    if args.model == 'decisiontreemodel':
        kwargs = {'use_reward_trees': args.use_reward_trees}
        if args.explorer_type in ['unvisitedbonusexplorer', 'leastvisitedbonusexplorer', 'unknownbonusexplorer']:
            kwargs.update({'explorer_type': args.explorer_type,
                           'rmax': args.rmax})
        if args.explorer_type == 'leastvisitedbonusexplorer' and args.thresh:
            kwargs.update({'thresh': args.thresh})
    else:
        args.ignore_unreachable = False
    model = MDPModelFactory.create(args.model, **kwargs)

    explorer = None
    if args.explorer_type in ['egreedyexplorer', 'softmaxexplorer']:
        explorer = ExplorerFactory.create(args.explorer_type, args.explorer_params, args.decay)

    if args.learner == 'apprenticeshiplearner':
        learner = None
        if args.progress:
            try:
                learner = ApprenticeshipLearner.load(args.savefile)
            except IOError:
                pass

        if not learner:
            try:
                data = load_from_file(args.infile)
                obs = data["state"]
                actions = data["act"]
                labels = data["label"]
            except IOError:
                sys.exit(sys.exc_info()[1])
            except KeyError, e:
                sys.exit("Key not found: {0}".format(e))

            # Train the model with empirical data
            for i, (s, a, l) in enumerate(zip(obs, actions, labels)):
                model.fit(s, a, l[0])
            # model.print_transitions()

            learner = ApprenticeshipLearner(np.asarray(demo),
                                            ValueIteration(model, explorer, args.gamma, args.ignore_unreachable),
                                            filename=args.savefile)

        learner.learn()

    else:
        a_args = (args.pip, args.pport, args.fsm_config,)
        a_args += ("learningmodule", args.keep_history, None, {"filename": args.infile, "append": args.append},
                   args.learner,np.asarray(demo), ValueIteration(model, explorer, args.gamma, args.ignore_unreachable),)
        a_kwargs = {'filename': args.savefile, 'progress': args.progress}

        client = None
        if args.client == 'webots':
            client = WebotsClient()
        
        run(PenaltyKickEnvironment, PenaltyKickAgent, EpisodicExperiment,
            env_args=(args.feature_rep, args.support_leg, client,),
            agent_args=a_args, agent_kwargs=a_kwargs,
            local=args.local)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Emulate the behavior via inverse reinforcement learning.")
    ap.add_argument("--local", action="store_false",
                    help="When true, the experiment is run through a local RLGlue implementation.")
    ap.add_argument("--progress", action="store_true", help="When true, a previously saved trial will be continued.")
    ap.add_argument("--keep_history", action="store_true",
                    help="When true, a history of states and actions performed by agent is saved to file <infile>.")
    ap.add_argument("--append", action="store_true",
                    help="When true, recorded data is appended to the file <infile>.")
    ap.add_argument("--use_reward_trees", action="store_true",
                    help="This option is only relevant for model type decision-tree. When true, decision trees "
                         "are used for the reward model.")
    ap.add_argument("--ignore_unreachable", action="store_true",
                    help="When true, unreachable states are being ignored during planning")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO IP address")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO port")
    ap.add_argument("--client", type=str, choices=["webots", "human"], default="webots", required=False,
                    help="The client the environment connects to.")
    ap.add_argument("--learner", type=str, choices=['apprenticeshiplearner', 'incrapprenticeshiplearner'],
                    default="apprenticeshiplearner", required=False, help="Learning type")
    ap.add_argument("--model", type=str, default="discretemodel", required=False, help="The model type")
    ap.add_argument("--explorer_type", type=str, default="leastvisitedbonusexplorer", required=False,
                    help="The explorer type type to use")
    ap.add_argument("--rmax", type=float, default=20.0, required=False, help="The maximum possible reward")
    ap.add_argument("--thresh", type=float, required=False, help="Model explorer threshold")
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
    ap.add_argument("--demofile", type=str, required=True, 
                    help="The name of the file containing the experts demonstration data.")
    ap.add_argument("--infile", type=str, required=False, help="Name of the file to store the recorded data to.")

    main(ap.parse_args())
