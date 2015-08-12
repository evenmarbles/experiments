import math

import numpy as np

from mlpy.experiments.task import EpisodicTask
from mlpy.mdp.stateaction import State, Action

from naobot.world_model.model import NaoWorldModel


class IRLPenaltyKickTask(EpisodicTask):
    """The penalty kick task.

    Implementation of the inverse reinforcement learning problem for the penalty kick task.
    This acts as the interface between the agent and the artificial intelligence.

    """
    BIN_WIDTH = 4

    def __init__(self, env=None):
        super(IRLPenaltyKickTask, self).__init__("ShiftWeight", ["Kick", "Fallen"], env)

        self._hip_roll = None
        """:type: str"""
        self._ankle_roll = None
        """:type: str"""

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

            s = [0] * State.nfeatures

            bin_num = int(math.floor((displacement - State.min_features) / IRLPenaltyKickTask.BIN_WIDTH))
            try:
                s[bin_num] = 1
            except IndexError, e:
                exit(e)
            s = State(s, name=state.name)
        return s

    def _configure_state(self):
        # noinspection PyShadowingNames
        def is_valid(self):
            num_ones = len(np.where(self.get()[0:len(self)] == 1)[0])
            if num_ones > 1 or num_ones < 1 or not all(i == 0 or i == 1 for i in self.get()):
                return False
            return True

        # noinspection PyShadowingNames
        def encode(self):
            return np.where(self.get()[0:len(self)] == 1)[0]

        def decode(cls, state_repr):
            decoded = [0] * cls.nfeatures
            bin_num = 0
            if isinstance(state_repr[0], int):
                bin_num = state_repr[0]
            elif isinstance(state_repr[0], float):
                bin_num = int(math.floor((state_repr[0] - cls.min_features) / IRLPenaltyKickTask.BIN_WIDTH))

            if 0 <= bin_num <= cls.nfeatures - 1:
                decoded[bin_num] = 1
            elif bin_num < 0:
                decoded[0] = 1
            else:
                decoded[cls.nfeatures - 1] = 1

            return cls(decoded)

        State.is_valid = is_valid
        State.encode = encode
        State.decode = classmethod(decode)

    def _configure_action(self):
        Action.set_description({
            "out": {"value": [-0.004]},
            "in": {"value": [0.004]},
            "kick": {"value": [-1.0]}
        })

    def _initialize(self):
        """Initializations."""
        min_hip_roll, max_hip_roll = NaoWorldModel().get_robot_info(self._hip_roll)
        leg_length = NaoWorldModel().get_robot_info("TibiaLength") + NaoWorldModel().get_robot_info("ThighLength")

        State.set_minmax_features(math.floor(leg_length * math.sin(min_hip_roll)),
                                  math.ceil(leg_length * math.sin(max_hip_roll)))
        State.set_nfeatures(int(math.ceil((State.max_features - State.min_features + 1) / IRLPenaltyKickTask.BIN_WIDTH)))
        State.set_description(['bin_'+str(e) for e in np.arange(State.nfeatures).tolist()])
