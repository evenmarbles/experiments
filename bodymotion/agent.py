import numpy as np

from rlglued.types import Observation
from rlglued.utils.taskspecvrlglue3 import TaskSpecParser


from mlpy.agents.modelbased import BotWrapper
from mlpy.agents.modelbased import ModelBasedAgent

from mlpy.mdp.stateaction import MDPState, MDPAction

from naobot.agent import NaoBot
from naobot.world_model.model import NaoWorldModel


class BodyMotionAgent(ModelBasedAgent):

    def __init__(self, pip, pport, fsm_config, module_type=None, record=False,
                 dataset_args=None, dataset_kwargs=None, *args, **kwargs):
        super(BodyMotionAgent, self).__init__(module_type, record, dataset_args, dataset_kwargs,
                                              BotWrapper(NaoBot(pip, pport, fsm_config)), *args, **kwargs)
        self._feature_rep = 'larm'

    def init(self, taskspec):
        super(BodyMotionAgent, self).init(taskspec)

        ts = TaskSpecParser(taskspec)
        if ts.valid:
            extra = ts.get_extra()

            v = ['FEATUREREP', 'STATESPERDIM', 'STATEDESCR', 'ACTIONDESCR', 'COPYRIGHT']
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

            MDPState.dtype = MDPState.DTYPE_INT

            if self._feature_rep == 'larm':
                def map_state_key(key):
                    return {
                        "x": 0,
                        "y": 1,
                        "z": 2,
                    }[key]

                def map_action_key(key):
                    return {
                        "dx": 0,
                        "dy": 1,
                        "dz": 2,
                    }[key]

            else:
                def map_state_key(key):
                    return {
                        "x": 0,
                        "y": 1,
                        "z": 2,
                        "wx": 3,
                        "wy": 4,
                        "wz": 5,
                    }[key]

                def map_action_key(key):
                    return {
                        "dx": 0,
                        "dy": 1,
                        "dz": 2,
                        "dwx": 3,
                        "dwy": 4,
                        "dwz": 5
                    }[key]

            MDPState.key_to_index = map_state_key
            MDPAction.key_to_index = map_action_key

    def start(self, observation):
        observation = self._update_observation()
        return super(BodyMotionAgent, self).start(observation)

    def step(self, reward, observation):
        observation = self._update_observation()
        return super(BodyMotionAgent, self).step(reward, observation)

    def obs2state(self, observation):
        return MDPState(observation, NaoWorldModel().get_fsm_state())

    def _update_observation(self):
        return_obs = Observation()

        return_obs.doubleArray = np.zeros(MDPState.nfeatures)
        for effector, mapping in MDPState.description.iteritems():
            pos = NaoWorldModel().get_effector_pos(effector)
            for key, axis in mapping.iteritems():
                return_obs.doubleArray[axis] = pos[MDPState.key_to_index(key)]
        return return_obs
