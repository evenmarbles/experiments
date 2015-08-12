import math

import numpy as np

from naobot.behavior import NaoState
from naobot.world_model.model import NaoWorldModel
from naobot.kinematics import NaoMotionController


def action_to_motion(action, support_leg):
    ankle_roll = "LAnkleRoll"
    hip_roll = "LHipRoll"

    if support_leg == "left":
        ankle_roll = "RAnkleRoll"
        hip_roll = "RHipRoll"

    ankle = NaoWorldModel().get_joint_pos(ankle_roll)[1:3]  # y,z position
    hip = NaoWorldModel().get_joint_pos(hip_roll)[1:3]  # y,z position

    yd = hip[0] - ankle[0]
    zd = hip[1] - ankle[1]
    delta = math.atan(yd / zd) - math.atan((yd + action) / zd)
    return NaoWorldModel().get_joint(hip_roll) + delta


class PenaltyKickState(NaoState):

    @property
    def support_leg(self):
        return self._support_leg

    @property
    def motion_completed(self):
        return self._motion_completed

    def __init__(self, motion, **kwargs):
        """
        Penalty Kick base state initialization.

        :type motion: NaoMotionController
        """
        super(PenaltyKickState, self).__init__(motion)

        self._support_leg = kwargs["support_leg"] if "support_leg" in kwargs else "right"
        self._motion_completed = False

    def enter(self, t, *args, **kwargs):
        super(PenaltyKickState, self).enter(t)
        self._motion_completed = False


class Idle(PenaltyKickState):
    def __init__(self, motion):
        """
        Fallen state initialization.

        :type motion: NaoMotionController
        """
        super(Idle, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        super(Idle, self).enter(t, *args, **kwargs)

    def update(self, dt):
        super(Idle, self).update(dt)


class Fallen(PenaltyKickState):
    def __init__(self, motion):
        """
        Fallen state initialization.

        :type motion: NaoMotionController
        """
        super(Fallen, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        super(Fallen, self).enter(t, *args, **kwargs)

    def update(self, dt):
        super(Fallen, self).update(dt)
        return "done"


class ShiftWeight(PenaltyKickState):
    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        """
        Shift weight state initialization.

        :type motion: NaoMotionController
        """
        super(ShiftWeight, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        """
        Enter the state.
        """
        super(ShiftWeight, self).enter(t, *args, **kwargs)

        self._motion.enter(t)

        # Shift weight to right leg and look down
        head_shift = [0.0, 0.0, 0.0, 0.0, 0.45553, 0.0]
        torso_shift = [0.0, -0.06, -0.03, 0.0, 0.0, 0.0]

        if self._support_leg == "left":
            torso_shift[1] *= -1

        delta_tf = [np.asarray(head_shift), np.asarray(torso_shift)]
        times = [[2.0], [2.0]]
        self._motion.transform(["Head", "Torso"], delta_tf, times)

    def update(self, dt):
        """
        Updates the state.
        """
        event = super(ShiftWeight, self).update(dt)
        if event is not None:
            return event

        if not self._motion.is_running():
            if not self._motion_completed:
                # Lift LLeg
                effectors = ["LLeg"]
                if self._support_leg == "left":
                    effectors = ["RLeg"]

                delta_tf = np.asarray([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
                times = [2.0]
                self._motion.transform(effectors, delta_tf, times, [NaoMotionController.FRAME_TORSO])

                self._motion_completed = True
                return


class ShiftOut(PenaltyKickState):
    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        """
        Shift out state initialization.

        :type motion: NaoMotionController
        """
        super(ShiftOut, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        """
        Shift kicking leg out.

        :param t: The current time (sec)
        :type t: float
        """
        super(ShiftOut, self).enter(t, *args, **kwargs)

        action = kwargs["action"]

        joints = "LHipRoll"
        if self._support_leg == "left":
            joints = "RHipRoll"
            action *= -1

        angle = action_to_motion(action.get(), self._support_leg)
        self._motion.interpolate_angles(joints, angle, 1.0)

    def update(self, dt):
        """
        Check whether the robot has fallen down.
        """
        event = super(ShiftOut, self).update(dt)
        if event is not None:
            return event

        if not self._motion.is_running():
            self._motion_completed = True


class ShiftIn(PenaltyKickState):
    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        """
        Shift in state initialization.

        :type motion: NaoMotionController
        """
        super(ShiftIn, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        """
        Shift the kicking leg in.
        """
        super(ShiftIn, self).enter(t, *args, **kwargs)

        action = kwargs["action"]

        joints = "LHipRoll"
        if self._support_leg == "left":
            joints = "RHipRoll"
            action *= -1

        angle = action_to_motion(action.get(), self._support_leg)
        self._motion.interpolate_angles(joints, angle, 1.0)

    def update(self, dt):
        """
        Check whether the robot has fallen down.
        """
        event = super(ShiftIn, self).update(dt)
        if event is not None:
            return event

        if not self._motion.is_running():
            self._motion_completed = True


class Kick(PenaltyKickState):
    # noinspection PyUnusedLocal
    def __init__(self, motion, **kwargs):
        """
        Kick state initialization.

        :type motion: NaoMotionController
        :type problem: IProblem
        :type cfg: dict[str, dict]
        """
        super(Kick, self).__init__(motion)

    def enter(self, t, *args, **kwargs):
        """
        Swinging the kicking leg back to gain momentum.
        """
        super(Kick, self).enter(t, *args, **kwargs)

        effectors = "LLeg"
        delta_pos = np.asarray([-0.1, 0.0, 0.0, 0.0, -0.03, 0.0])

        if self._support_leg == "left":
            effectors = "RLeg"
            delta_pos[4] *= -1

        times = [2.0]
        self._motion.interpolate_positions(effectors, delta_pos, times, NaoMotionController.FRAME_TORSO)

    def update(self, dt):
        """
        Swing the kicking leg forward quickly to reach maximum speed
        of the ball.
        """
        event = super(Kick, self).update(dt)
        if event is not None:
            return event

        if not self._motion.is_running():
            if not self._motion_completed:
                effectors = "LLeg"
                delta_pos1 = np.asarray([0.15, 0.0, 0.0, 0.0, -0.03, 0.0])

                if self._support_leg == "left":
                    effectors = "RLeg"
                    delta_pos1[4] *= -1

                delta_pos = [[delta_pos1,
                             np.asarray([0.09, 0.0, 0.00, 0.0, 0.0, 0.0])]]
                times = [[0.2, 0.3]]
                self._motion.interpolate_positions(effectors, delta_pos, times, NaoMotionController.FRAME_TORSO)

                self._motion_completed = True
                return

            return "done"

    def exit(self):
        super(Kick, self).exit()

        self._motion.go_to_default_posture()
