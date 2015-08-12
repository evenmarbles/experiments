import math

import numpy as np

from mlpy.experiments.task import EpisodicTask
from mlpy.mdp.stateaction import State, Action
from mlpy.constants import eps

from naobot.world_model.model import NaoWorldModel


class RLPenaltyKickTask(EpisodicTask):
    """
    Implementation of the inverse reinforcement learning problem for the penalty kick task.
    This acts as the interface between the agent and the artificial intelligence.
    """

    def __init__(self, env=None):
        super(RLPenaltyKickTask, self).__init__("ShiftWeight", ["Kick", "Fallen"], env)

        self._ball = NaoWorldModel().get_object("ball")

        self._hip_roll = None
        """:type: str"""
        self._ankle_roll = None
        """:type: str"""

        self._rewards = {
            "states": {
                "Fallen": -20
            },
            "actions": {
                "in": -1,
                "out": -1,
                "kick": {
                    "success": 20,
                    "failure": -2
                }
            }
        }

        self._event_delay_on_term = 5000

    def sensation(self, state):
        """Gather the state feature information.

        Gather the state information (i.e. features) according to
        the task from the agent's senses.

        Parameters
        ----------
        kwargs: dict
            Non-positional arguments needed for gathering the
            information.

        Returns
        -------
        features : array, shape (`nfeatures`,)
            The sensed features.

        """
        if self._hip_roll is None and self._ankle_roll is None:
            self._ankle_roll = "LAnkleRoll"
            self._hip_roll = "LHipRoll"
            if state.support_leg == "left":
                self._ankle_roll = "RAnkleRoll"
                self._hip_roll = "RHipRoll"

            self._initialize()

        s = None
        if state.name not in ["Idle"]:
            y_ankle = NaoWorldModel().get_joint_pos(self._ankle_roll)[1]
            y_hip = NaoWorldModel().get_joint_pos(self._hip_roll)[1]
            displacement = (y_ankle - y_hip) * 1000  # convert to millimeters

            image_x = self._ball.image_center.x
            if image_x == 0:
                return None

            s = State(np.asarray([image_x, displacement]), name=state.name)
        return s

    def get_reward(self, state, action):
        """Retrieve the reward.

        Retrieve the reward for the given state and action from
        the environment.

        Parameters
        ----------
        state : State
            The current state.
        action : Action
            The current action.

        Returns
        -------
        float :
            The reward.

        """
        if state is not None and action is not None:
            for key in self._rewards:
                for event in self._rewards[key]:
                    if key == "states":
                        if state.name == event:
                            return self._read_reward(self._rewards[key][event])
                    elif key == "actions":
                        if action.name == event:
                            return self._read_reward(self._rewards[key][event])
            assert False, "No reward defined for state-action pair: {}-{}".format(state, action)

    def _configure_state(self):
        # noinspection PyShadowingNames
        def is_valid(self):
            real_state = True

            if State.min_features is not None:
                for (feature, min_feature, max_feature) in zip(self, State.min_features, State.max_features):
                    if feature < (min_feature - eps) or feature > (max_feature + eps):
                        real_state = False
                        self._logger.debug("\t\t\t\tNext state is not valid (feature %d out of range)", feature)
                        break

            return real_state
        State.is_valid = is_valid

    def _configure_action(self):
        Action.set_description({
            "out": {"value": [-0.004]},
            "in": {"value": [0.004]},
            "kick": {"value": [-1.0]}
        })

    def _initialize(self):
        """
        Initializations.
        """
        min_hip_roll, max_hip_roll = NaoWorldModel().get_robot_info(self._hip_roll)
        leg_length = NaoWorldModel().get_robot_info("TibiaLength") + NaoWorldModel().get_robot_info("ThighLength")

        try:
            max_location = self._ball.resolution[0]
        except AttributeError:
            max_location = 0

        State.set_minmax_features([0, math.floor(leg_length * math.sin(min_hip_roll))],
                                  [max_location, math.ceil(leg_length * math.sin(max_hip_roll))])
        State.set_states_per_dim([int((State.max_features[0] - State.min_features[0]) / 2),
                                  int(math.ceil((State.max_features[1] - State.min_features[1]) / 4))])
        State.set_description({
            "descr": {
                "image x-position": 0,
                "displacement (mm)": 1
            }
        })

    def _read_reward(self, config):
        """Read the reward according to the configuration.

        Parameters
        ----------
        config : dict
            The configuration.

        """
        if type(config) == int:
            return config

        if type(config) == dict:
            kick_result = self._env.check_data("check goal")
            return config[kick_result]
