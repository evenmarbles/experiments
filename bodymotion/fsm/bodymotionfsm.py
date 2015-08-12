import numpy as np

from mlpy.mdp.stateaction import Action

from naobot.behavior import NaoState
from naobot.kinematics import NaoMotionController


class BodyMotionState(NaoState):

    @property
    def motion_completed(self):
        return self._motion_completed

    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        """
        Penalty Kick base state initialization.

        :type motion: NaoMotionController
        """
        super(BodyMotionState, self).__init__(motion)

        self._motion_completed = False

    def enter(self, t, *args, **kwargs):
        super(BodyMotionState, self).enter(t)
        self._motion_completed = False


class WakeUp(BodyMotionState):

    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        super(WakeUp, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        super(WakeUp, self).enter(t, *args, **kwargs)

        self._motion.enter(t)

    def update(self, dt):
        super(WakeUp, self).update(dt)

        if not self._motion.is_running():
            self._motion_completed = True


class MoveEffectors(BodyMotionState):

    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        super(MoveEffectors, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        super(MoveEffectors, self).enter(t, *args, **kwargs)

        action = kwargs["action"]
        config = action.description[action.name]

        effectors = []
        times = []
        delta_tf = []
        frames = []

        for effector, mapping in config["descr"].iteritems():
            delta = np.zeros(len(mapping.keys()))
            for key, axis in mapping.iteritems():
                delta[Action.key_to_index(key)] = action[axis]

            if np.any(delta):
                effectors.append(effector)
                delta_tf.append(delta)
                times.append([0.5])
                frames.append(NaoMotionController.FRAME_TORSO)

        self._motion.transform(effectors, delta_tf, times, frames)

    def update(self, dt):
        super(MoveEffectors, self).update(dt)

        if not self._motion.is_running():
            self._motion_completed = True
