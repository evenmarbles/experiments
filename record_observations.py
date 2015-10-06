import sys
import argparse

from mlpy.auxiliary.io import load_from_file
from mlpy.experiments.run import run
from mlpy.experiments.episodic import EpisodicExperiment
from mlpy.environments.utils.webots.client import WebotsClient
from mlpy.tools.misc import convert_to_policy

from penaltykick.environment import PenaltyKickEnvironment
from penaltykick.agent import PenaltyKickAgent
from bodymotion.environment import BodyMotionEnvironment
from bodymotion.agent import BodyMotionAgent


def main(args):
    try:
        filename = convert_to_policy(args.policies)
        # noinspection PyUnusedLocal
        data = load_from_file(filename)
        if ":" not in args.policy_num:
            args.policy_num = args.policy_num + ":" + str(int(args.policy_num) + 1)
        policies = eval("data['act'][" + str(args.policy_num) + "]")
    except IOError:
        sys.exit(sys.exc_info()[1])
    except KeyError, e:
        sys.exit("Key not found: {0}".format(e))

    a_args = (args.pip, args.pport, args.fsm_config,)
    a_args += ("followpolicymodule", True, None, {"filename": args.file, "append": args.append},
               policies,)

    client = None
    if args.client == 'webots':
        client = WebotsClient()

    if args.exp == "penaltykick":
        run(PenaltyKickEnvironment, PenaltyKickAgent, EpisodicExperiment,
            env_args=(args.feature_rep, args.support_leg, client,),
            agent_args=a_args,
            exp_kwargs={'ntrials': args.ntrials, 'nepisodes': args.nepisodes},
            local=args.local)
    elif args.exp == "bodymotion":
        run(BodyMotionEnvironment, BodyMotionAgent, EpisodicExperiment,
            env_args=(args.feature_rep, client,),
            agent_args=a_args,
            exp_kwargs={'ntrials': args.ntrials, 'nepisodes': args.nepisodes},
            local=args.local)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gather observation data for the given task from the NAO robot.")
    ap.add_argument("--local", action="store_false",
                    help="When true, the experiment is run through a local RLGlue implementation.")
    ap.add_argument("--append", action="store_true",
                    help="When true, observations are appended to the file <file>.")
    ap.add_argument("--exp", type=str, default="penaltykick", choices=["penaltykick", "bodymotion"], required=False,
                    help="The experiment to run")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO IP address.")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO port.")
    ap.add_argument("--client", type=str, choices=["webots", "human"], default="webots", required=False,
                    help="The client the environment connects to.")
    ap.add_argument("--ntrials", type=int, default=1, required=False,
                    help="The number of trials to run.")
    ap.add_argument("--nepisodes", type=int, default=1, required=False,
                    help="The number of episodes to run in each trial.")
    ap.add_argument("--policy_num", type=str, default=0, required=False,
                    help="The identification of the policies to run.")
    ap.add_argument("--support_leg", type=str, default="right", required=False,
                    help="The supporting leg. The information must match with the supporting leg in "
                         "the FSM configuration")
    ap.add_argument("--feature_rep", type=str, choices=["rl", "irl", "larm", "wholebody"], default="rl", required=False,
                    help="The feature representation to use for the experiment.")
    ap.add_argument("--fsm_config", type=str, required=True, help="FSM configuration file.")
    ap.add_argument("--policies", type=str, required=True, help="The name of the file containing the policies.")
    ap.add_argument("--file", type=str, required=True, help="Name of the file to store the recorded data to.")

    main(ap.parse_args())
