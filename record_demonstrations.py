import argparse

from mlpy.tools.configuration import ConfigMgr
from mlpy.environments.nao import NaoEnvFactory
from mlpy.experiments import Experiment

from naobot.agent import NaoBot

from taskfactory import TaskFactory


def main(args):
    env = NaoEnvFactory.create(args.env, args.sport)

    agent = NaoBot(args.pip, args.pport, "usermodule", TaskFactory.create(args.task),
                   args.fsm_config, True, {"filename": args.file, "append": args.append},
                   ConfigMgr(args.user_config, "pygame", eval_key=True), args.ndemo)

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
    ap.add_argument("--ndemo", type=int, default=1, required=False, help="Number of demonstration.")
    ap.add_argument("--fsm_config", type=str, required=True, help="FSM configuration file.")
    ap.add_argument("--user_config", type=str, required=True, help="Events to action configuration.")
    ap.add_argument("--file", type=str, required=True, help="Demonstration data.")
    ap.add_argument("--task", type=str, required=True, help="The task to perform.")

    main(ap.parse_args())
