from penaltykick.irl import IRLPenaltyKickTask
from penaltykick.rl import RLPenaltyKickTask
from bodymotion import WholeBodyMotionTask
from bodymotion import LarmBodyMotionTask


class TaskFactory(object):

    @staticmethod
    def create(_type, *args, **kwargs):
        try:
            return {
                "penaltykick-irl": IRLPenaltyKickTask,
                "penaltykick-rl": RLPenaltyKickTask,
                "larmbodymotion-casml": LarmBodyMotionTask,
                "wholebodymotion-casml": WholeBodyMotionTask,
            }[_type](*args, **kwargs)

        except KeyError:
            return None
