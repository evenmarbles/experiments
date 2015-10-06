import math
import numpy as np

from rlglued.types import Observation
from rlglued.utils.taskspecvrlglue3 import TaskSpecParser

from mlpy.agents.modelbased import BotWrapper
from mlpy.agents.modelbased import ModelBasedAgent
from mlpy.mdp.stateaction import MDPState
from mlpy.constants import eps

from naobot.agent import NaoBot
from naobot.world_model.model import NaoWorldModel


class PenaltyKickAgent(ModelBasedAgent):
    def __init__(self, pip, pport, fsm_config, module_type=None, record=False,
                 dataset_args=None, dataset_kwargs=None, *args, **kwargs):
        super(PenaltyKickAgent, self).__init__(module_type, record, dataset_args, dataset_kwargs,
                                               BotWrapper(NaoBot(pip, pport, fsm_config)), *args, **kwargs)
        self._feature_rep = 'rl'
        self._support_leg = 'right'
        self._bin_width = 4

        self._ankle_roll = "LAnkleRoll"
        self._hip_roll = "LHipRoll"

    def init(self, taskspec):
        super(PenaltyKickAgent, self).init(taskspec)

        ts = TaskSpecParser(taskspec)
        if ts.valid:
            extra = ts.get_extra()

            v = ['FEATUREREP', 'SUPPORTLEG', 'STATEDESCR', 'ACTIONDESCR', 'COPYRIGHT']
            pos = []
            for i, id_ in enumerate(list(v)):
                try:
                    pos.append(extra.index(id_))
                except:
                    v.remove(id_)
            sorted_v = sorted(zip(pos, v))
            v = [s[1] for s in sorted_v]

            for i, id_ in enumerate(v):
                val = ts.get_value(i, extra, v)
                if id_ == 'FEATUREREP':
                    self._feature_rep = val
                if id_ == 'SUPPORTLEG':
                    if val == 'left':
                        self._ankle_roll = "RAnkleRoll"
                        self._hip_roll = "RHipRoll"

            min_hip_roll, max_hip_roll = NaoWorldModel().get_robot_info(self._hip_roll)
            leg_length = NaoWorldModel().get_robot_info("TibiaLength") + NaoWorldModel().get_robot_info("ThighLength")

            MDPState.dtype = MDPState.DTYPE_INT

            if self._feature_rep == 'rl':
                try:
                    max_location = NaoWorldModel().get_object("ball").resolution[0]
                except AttributeError:
                    max_location = 0

                MDPState.set_minmax_features([0, max_location],
                                             [math.floor(leg_length * math.sin(min_hip_roll)),
                                              math.ceil(leg_length * math.sin(max_hip_roll))])
                MDPState.set_states_per_dim([int((MDPState.max_features[0] - MDPState.min_features[0]) / 2),
                                             int(math.ceil((MDPState.max_features[1] - MDPState.min_features[1]) / 4))])

                # noinspection PyShadowingNames
                def is_valid(self):
                    real_state = True

                    if MDPState.min_features is not None:
                        for (feature, min_feature, max_feature) in zip(self, MDPState.min_features,
                                                                       MDPState.max_features):
                            if feature < (min_feature - eps) or feature > (max_feature + eps):
                                real_state = False
                                self._logger.debug("\t\t\t\tNext state is not valid (feature %d out of range)", feature)
                                break

                    return real_state

                MDPState.is_valid = is_valid

            else:
                MDPState.set_minmax_features([math.floor(leg_length * math.sin(min_hip_roll)),
                                              math.ceil(leg_length * math.sin(max_hip_roll))])
                MDPState.set_nfeatures(
                    int(math.ceil((MDPState.max_features - MDPState.min_features + 1) / self._bin_width)))

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
                        bin_num = int(math.floor((state_repr[0] - cls.min_features) / self._bin_width))

                    if 0 <= bin_num <= cls.nfeatures - 1:
                        decoded[bin_num] = 1
                    elif bin_num < 0:
                        decoded[0] = 1
                    else:
                        decoded[cls.nfeatures - 1] = 1

                    return cls(decoded)

                MDPState.is_valid = is_valid
                MDPState.encode = encode
                MDPState.decode = decode

    def start(self, observation):
        observation = self._update_observation()
        return_action = super(PenaltyKickAgent, self).start(observation)
        if NaoWorldModel().has_fallen:
            return_action.doubleArray = [-2.0]
        return return_action

    def step(self, reward, observation):
        observation = self._update_observation()
        return_action = super(PenaltyKickAgent, self).step(reward, observation)
        if NaoWorldModel().has_fallen:
            return_action.doubleArray = [-2.0]
        return return_action

    def obs2state(self, observation):
        return MDPState(observation, NaoWorldModel().get_fsm_state())

    def _update_observation(self):
        return_obs = Observation()

        y_ankle = NaoWorldModel().get_joint_pos(self._ankle_roll)[1]
        y_hip = NaoWorldModel().get_joint_pos(self._hip_roll)[1]
        displacement = (y_ankle - y_hip) * 1000  # convert to millimeters

        if self._feature_rep == 'rl':
            image_x = NaoWorldModel().get_object("ball").image_center.x
            return_obs.intArray = [image_x, displacement]
        else:
            s = [0] * MDPState.nfeatures

            bin_num = int(math.floor((displacement - MDPState.min_features) / self._bin_width))
            try:
                s[bin_num] = 1
            except IndexError, e:
                exit(e)
            return_obs.intArray = s

        return return_obs
