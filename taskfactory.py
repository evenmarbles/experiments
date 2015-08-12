from penaltykick.irl import IRLPenaltyKickTask
from penaltykick.rl import RLPenaltyKickTask
from bodymotion import BodyMotionTask


class TaskFactory(object):

    @staticmethod
    def create(_type, *args, **kwargs):
        try:
            return {
                "penaltykick-irl": IRLPenaltyKickTask,
                "penaltykick-rl": RLPenaltyKickTask,
                "bodymotion-casml": BodyMotionTask,
            }[_type](*args, **kwargs)

        except KeyError:
            return None
