from rlglued.environment import loader
from rlglued.utils.taskspecvrlglue3 import TaskSpec
from rlglued.environment.environment import Environment
from rlglued.types import Observation
from rlglued.types import Reward_observation_terminal


# /**
#  *  This is a very simple environment with discrete observations corresponding to states labeled {0,1,...,19,20}
#     The starting state is 10.
# 
#     There are 2 actions = {0,1}.  0 decrements the state, 1 increments the state.
# 
#     The problem is episodic, ending when state 0 or 20 is reached, giving reward -1 or +1, respectively.
# 	  The reward is 0 on all other steps.
#  * @author Brian Tanner
#  */

class BodyMotionEnvironment(Environment):

    def __init__(self, feature_rep='larm', client=None):
        super(BodyMotionEnvironment, self).__init__()

        self._client = client

        if feature_rep not in ['larm', 'wholebody']:
            raise ValueError("Unknown feature representation '%s'." % feature_rep)
        self._feature_rep = feature_rep

        self._ts_extra = "FEATUREREP %s " % feature_rep

        if feature_rep == 'larm':
            state_descr = "{'LArm':{'x':0,'y':1,'z':2}}"
            action_desc = "{'move':{'value':'*','descr':{'LArm':{'dx':0,'dy':1,'dz':2}}}}"
        else:
            state_descr = "{'LArm':{'x':0,'y':1,'z':2},'RArm':{'x':3,'y':4,'z':5},'LLeg':{'x':6,'y':7,'z':8}," \
                          "'RLeg':{'x':9,'y':10,'z':11},'Torso':{'x':12,'y':13,'z':14}}"
            action_desc = "{'move':{'value':'*','descr':{'LArm':{'dx':0,'dy':1,'dz':2},'RArm':{'dx':3,'dy':4,'dz':5}," \
                          "'LLeg':{'dx':6,'dy':7,'dz':8},'RLeg':{'dx':9,'dy':10,'dz':11}," \
                          "'Torso':{'dx':12,'dy':13,'dz':14}}}}"

        self._ts_extra += "STATEDESCR %s " % state_descr
        self._ts_extra += "ACTIONDESCR %s " % action_desc

    def init(self):
        ts = TaskSpec(discount_factor=0.99, reward_range=(-20, 20))
        ts.set_continuing()
        ts.set_charcount_obs(0)
        ts.add_double_act((0, 2))

        ts.set_extra(self._ts_extra + "COPYRIGHT Penaltykick (Python) implemented by Astrid Jackson")
        return ts.to_taskspec()

    def setup(self):
        if self._client is not None:
            self._client.reset()

    def start(self):
        return Observation()

    def step(self, action):
        return_ro = Reward_observation_terminal()
        return_ro.r = -1.0
        return_ro.o = Observation()
        return_ro.terminal = False

        return return_ro

    def cleanup(self):
        if self._client is not None:
            self._client.close()

    def message(self, msg):
        if msg == "what is your name?":
            return "my name is skeleton_environment, Python edition!"
        else:
            return "I don't know how to respond to your message"


if __name__ == "__main__":
    loader.load_environment(BodyMotionEnvironment())
