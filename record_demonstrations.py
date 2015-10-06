import argparse

from mlpy.experiments.run import run
from mlpy.experiments.episodic import EpisodicExperiment
from mlpy.tools.configuration import ConfigMgr
from mlpy.environments.utils.webots.client import WebotsClient

from penaltykick.environment import PenaltyKickEnvironment
from penaltykick.agent import PenaltyKickAgent
from bodymotion.environment import BodyMotionEnvironment
from bodymotion.agent import BodyMotionAgent


def main(args):
    a_args = (args.pip, args.pport, args.fsm_config,)
    a_args += ("usermodule", True, None, {"filename": args.file, "append": args.append},
               ConfigMgr(args.user_config, "pygame", eval_key=True),)

    client = None
    if args.client == 'webots':
        client = WebotsClient()

    if args.exp == "penaltykick":
        run(PenaltyKickEnvironment, PenaltyKickAgent, EpisodicExperiment,
            env_args=(args.feature_rep, args.support_leg, client,),
            agent_args=a_args,
            exp_kwargs={'nepisodes': args.nepisodes},
            local=args.local)
    elif args.exp == "bodymotion":
        run(BodyMotionEnvironment, BodyMotionAgent, EpisodicExperiment,
            env_args=(args.feature_rep, client,),
            agent_args=a_args,
            exp_kwargs={'nepisodes': args.nepisodes},
            local=args.local)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gather observation data for the given feature representation "
                                             "from the NAO robot.")
    ap.add_argument("--local", action="store_true",
                    help="When provided, the experiment is run through a local RLGlue implementation.")
    ap.add_argument("--append", action="store_true",
                    help="When provided, recorded data is appended to the file <file>.")
    ap.add_argument("--exp", type=str, default="penaltykick", choices=["penaltykick", "bodymotion"], required=False,
                    help="The experiment to run")
    ap.add_argument("--pip", type=str, default="127.0.0.1", required=False, help="NAO's IP address.")
    ap.add_argument("--pport", type=int, default=9559, required=False, help="NAO's port.")
    ap.add_argument("--client", type=str, choices=["webots", "human"], default="webots", required=False,
                    help="The client the environment connects to.")
    ap.add_argument("--nepisodes", type=int, default=1, required=False,
                    help="The number of episodes to run in each trial.")
    ap.add_argument("--support_leg", type=str, default="right", required=False,
                    help="The supporting leg. The information must match with the supporting leg in "
                         "the FSM configuration")
    ap.add_argument("--feature_rep", type=str, choices=["rl", "irl", "larm", "wholebody"], default="rl", required=False,
                    help="The feature representation to use for the experiment.")
    ap.add_argument("--fsm_config", type=str, required=True,
                    help="Name of the file containing the finite state machine configuration.")
    ap.add_argument("--user_config", type=str, required=True,
                    help="Name of the file containing the configuration to convert events to actions.")
    ap.add_argument("--file", type=str, required=True, help="Name of the file to store the recorded data to.")

    main(ap.parse_args())
