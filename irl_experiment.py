import sys
import argparse

import numpy as np

from mlpy.auxiliary.io import load_from_file
from mlpy.environments.nao import NaoEnvFactory
from mlpy.experiments import Experiment
from mlpy.mdp import MDPModelFactory
from mlpy.planners.explorers import ExplorerFactory
from mlpy.planners.discrete import ValueIteration
from mlpy.learners import ApprenticeshipLearner

from naobot.agent import NaoBot

from taskfactory import TaskFactory


def main(args):
    try:
        data = load_from_file(args.demofile)
        demo = data["state"]
    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    # noinspection PyUnusedLocal
    task = TaskFactory.create(args.task)

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
        env = NaoEnvFactory.create(args.env, args.sport)

        agent = NaoBot(args.pip, args.pport, "learningmodule", task,
                       args.fsm_config, args.keep_history, {"filename": args.infile, "append": args.append},
                       args.learner, np.asarray(demo),
                       ValueIteration(model, explorer, args.gamma, args.ignore_unreachable),
                       filename=args.savefile, progress=args.progress)

        env.add_agents(agent)

        experiment = Experiment(env)
        experiment.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Emulate the behavior via inverse reinforcement learning.")
    ap.add_argument("--progress", action="store_true", help="When true, a previously saved run will be continued")
    ap.add_argument("--keep_history", action="store_true",
                    help="When true, a history of states and actions performed by agent is saved to file")
    ap.add_argument("--append", action="store_true",
                    help="When true, observations are appended to the <file>.")
    ap.add_argument("--use_reward_trees", action="store_true",
                    help="This option is only relevant for model type decision-tree. When true, decision trees "
                         "are used for the reward model.")
    ap.add_argument("--ignore_unreachable", action="store_true",
                    help="When true, unreachable states are being ignored during planning")
    ap.add_argument("--env", type=str, default="nao.webots", required=False,
                    help="The environment in which to run the experiment")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO IP address")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO port")
    ap.add_argument("--sport", type=int, default=12345, required=False, help="Webots supervisor port")
    ap.add_argument("--task", type=str, choices=['penaltykick-irl', 'bodymotion-casml'], default="penaltykick-irl",
                    required=False, help="The task type")
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
    ap.add_argument("--fsm_config", type=str, required=True, help="FSM configuration file.")
    ap.add_argument("--savefile", type=str, required=False, help="The learner run file name.")
    ap.add_argument("--demofile", type=str, required=True, help="The experts demonstration data file name.")
    ap.add_argument("--infile", type=str, required=False, help="The trajectory data file name.")

    main(ap.parse_args())
