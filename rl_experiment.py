import argparse

from mlpy.environments.nao import NaoEnvFactory
from mlpy.experiments import Experiment
from mlpy.mdp import MDPModelFactory
from mlpy.planners.explorers import ExplorerFactory
from mlpy.planners.discrete import ValueIteration

from naobot.agent import NaoBot

from taskfactory import TaskFactory


def main(args):
    # First configure State and Action
    env = NaoEnvFactory.create(args.env, args.sport)
    task = TaskFactory.create(args.task, env)

    explorer = None
    if args.explorer_type in ['egreedyexplorer', 'softmaxexplorer']:
        explorer = ExplorerFactory.create(args.explorer_type, args.explorer_params, args.decay)

    if args.learner == 'rldtlearner':
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

        agent = NaoBot(args.pip, args.pport, "learningmodule", task,
                       args.fsm_config, args.keep_history, {"filename": args.file, "append": args.append},
                       args.learner, ValueIteration(model, explorer, args.gamma, args.ignore_unreachable),
                       filename=args.savefile, progress=args.progress)
    else:
        agent = NaoBot(args.pip, args.pport, "learningmodule", task,
                       args.fsm_config, args.keep_history, {"filename": args.file, "append": args.append},
                       args.learner, explorer, filename=args.savefile, progress=args.progress)

    env.add_agents(agent)

    experiment = Experiment(env)
    experiment.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment: Learn policy via reinforcement learning.")
    ap.add_argument("--progress", action="store_true", help="When true, a previously saved run will be continued")
    ap.add_argument("--keep_history", action="store_true",
                    help="When true, a history of states and actions performed by agent is saved to file")
    ap.add_argument("--append", action="store_true",
                    help="When true, observations are appended to the <file>.")
    ap.add_argument("--ignore_unreachable", action="store_true",
                    help="When true, unreachable states are being ignored during planning")
    ap.add_argument("--env", type=str, default="nao.webots", required=False,
                    help="The environment in which to run the experiment")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO IP address")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO port")
    ap.add_argument("--sport", type=int, default=12345, required=False, help="Webots supervisor port")
    ap.add_argument("--task", type=str, choices=['penaltykick-rl'], default="penaltykick-rl", required=False,
                    help="The task type")
    ap.add_argument("--learner", type=str, choices=['qlearner', 'rldtlearner'], default="rldtlearner", required=False,
                    help="Learning type")
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
    ap.add_argument("--fsm_config", type=str, required=True, help="FSM configuration file.")
    ap.add_argument("--savefile", type=str, required=False, help="The learner run file name")
    ap.add_argument("--file", type=str, required=False, help="Recorded data")

    main(ap.parse_args())
