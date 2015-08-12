import numpy as np

from mlpy.mdp.stateaction import State, Action
from mlpy.experiments.task import Task

from naobot.world_model.model import NaoWorldModel


class BodyMotionTask(Task):
    """The body motion task.

    """
    def __init__(self):
        super(BodyMotionTask, self).__init__()

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
                "wx": 3,
                "wy": 4,
                "wz": 5,
            }[key]

        State.key_to_index = staticmethod(map_state_key)

        State.set_description({
            "LArm": {"x": 0, "y": 1, "z": 2},
            "RArm": {"x": 3, "y": 4, "z": 5},
            "LLeg": {"x": 6, "y": 7, "z": 8},
            "RLeg": {"x": 9, "y": 10, "z": 11},
            "Torso": {"x": 12, "y": 13, "z": 14}
        })

    def _configure_action(self):
        # noinspection PyDecorator
        @staticmethod
        def map_action_key(key):
            return {
                "dx": 0,
                "dy": 1,
                "dz": 2,
                "dwx": 3,
                "dwy": 4,
                "dwz": 5
            }[key]
        Action.key_to_index = map_action_key

        Action.set_description({
            "move": {
                "value": "*",
                "descr": {
                    "LArm": {"dx": 0, "dy": 1, "dz": 2},
                    "RArm": {"dx": 3, "dy": 4, "dz": 5},
                    "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                    "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                    "Torso": {"dx": 12, "dy": 13, "dz": 14}
                }
            }
        })
