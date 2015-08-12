import sys
import argparse
import numpy as np

from mlpy.auxiliary.io import load_from_file, is_pickle, txt2pickle
from mlpy.environments.nao import NaoEnvFactory
from mlpy.experiments import Experiment
from mlpy.mdp.stateaction import Action

from naobot.agent import NaoBot

from taskfactory import TaskFactory


def convert_to_policy(filename):
    """
    Converts the list of floats into a list of policies of actions.

    Parameters
    ----------
    filename: str
        Name of the file containing the list of floats.

    Returns
    -------
    str :
        Name of the file to which the policies have been saved to.

    """
    if not is_pickle(filename):
        def convert(d):
            arr = np.zeros((len(d),), dtype=np.object)
            for i, l in enumerate(d):
                line = eval(l)
                size = len(line) if hasattr(line, '__len__') else 1
                s = np.zeros((Action.nfeatures, size), dtype=Action.dtype)
                if size > 1:
                    for j, v in enumerate(line):
                        s[:, j] = np.asarray(v)
                else:
                    s[0] = line
                arr[i] = s
            return {'act': arr, 'act_desc': Action.description}

        return txt2pickle(filename, func=convert)


def main(args):
    # First configure State and Action
    env = NaoEnvFactory.create(args.env, args.sport)
    task = TaskFactory.create(args.task, env)

    try:
        filename = convert_to_policy(args.policy)
        # noinspection PyUnusedLocal
        data = load_from_file(filename)
        if ":" not in args.policy_num:
            args.policy_num = args.policy_num + ":" + str(int(args.policy_num) + 1)
        policies = eval("data['act'][" + str(args.policy_num) + "]")
    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    agent = NaoBot(args.pip, args.pport, "followpolicymodule", task,
                   args.fsm_config, True, {"filename": args.file, "append": args.append},
                   policies, args.niter)
    env.add_agents(agent)

    experiment = Experiment(env)
    experiment.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gather observation data for the given task from the NAO robot.")
    ap.add_argument("--append", action="store_true",
                    help="When true, observations are appended to the <file>.")
    ap.add_argument("--pip", default="127.0.0.1", type=str, required=False, help="NAO IP address.")
    ap.add_argument("--pport", default=9559, type=int, required=False, help="NAO port.")
    ap.add_argument("--sport", type=int, default=12345, required=False, help="Webots supervisor port.")
    ap.add_argument("--env", type=str, default="nao.webots", required=False,
                    help="The environment in which to run the experiment.")
    ap.add_argument("--policy_num", type=str, default=0, required=False,
                    help="The identification of the policies to run.")
    ap.add_argument("--niter", type=int, default=1, required=False, help="Number of iterations per policy.")
    ap.add_argument("--fsm_config", type=str, required=True, help="FSM configuration file.")
    ap.add_argument("--policy", type=str, required=True, help="Policy file.")
    ap.add_argument("--file", type=str, required=True, help="Observation data.")
    ap.add_argument("--task", type=str, required=True, help="The task to perform.")

    main(ap.parse_args())
