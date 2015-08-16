import numpy as np

from mlpy.mdp.stateaction import State, Action
from mlpy.experiments.task import Task

from naobot.world_model.model import NaoWorldModel


class LarmBodyMotionTask(Task):
    """The body motion task.

    """
    def __init__(self, env=None):
        super(LarmBodyMotionTask, self).__init__(env)

    def sensation(self, **kwargs):
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
        s = np.zeros(State.nfeatures)
        for effector, mapping in State.description.iteritems():
            pos = NaoWorldModel().get_effector_pos(effector)
            for key, axis in mapping.iteritems():
                s[axis] = pos[State.key_to_index(key)]
        return State(s)

    def _configure_state(self):
        def map_state_key(key):
            return {
                "x": 0,
                "y": 1,
                "z": 2,
            }[key]
        State.key_to_index = staticmethod(map_state_key)

        State.set_description({
            "LArm": {"x": 0, "y": 1, "z": 2}
        })

    def _configure_action(self):
        def map_action_key(key):
            return {
                "dx": 0,
                "dy": 1,
                "dz": 2,
            }[key]
        Action.key_to_index = staticmethod(map_action_key)

        Action.set_description({
            "move": {
                "value": "*",
                "descr": {
                    "LArm": {"dx": 0, "dy": 1, "dz": 2}
                }
            }
        })
