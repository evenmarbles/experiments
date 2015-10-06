from rlglued.environment import loader
from rlglued.utils.taskspecvrlglue3 import TaskSpec
from rlglued.environment.environment import Environment
from rlglued.types import Observation
from rlglued.types import Reward_observation_terminal

from mlpy.mdp.stateaction import MDPAction


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

class PenaltyKickEnvironment(Environment):
    def __init__(self, feature_rep='rl', support_leg='right', client=None):
        super(PenaltyKickEnvironment, self).__init__()

        self._client = client

        if feature_rep not in ['rl', 'irl']:
            raise ValueError("Unknown feature representation '%s'." % feature_rep)
        self._feature_rep = feature_rep

        self._ts_extra = "FEATUREREP " + feature_rep + " SUPPORTLEG " + support_leg + " "

        if feature_rep == 'rl':
            state_descr = "{'descr':{'image x-position':0,'displacement (mm)':1}}"
        else:
            state_descr = "['bin_' + str(e) for e in np.arange(MDPState.nfeatures).tolist()]"

        self._ts_extra += "STATEDESCR %s " % state_descr

    def init(self):
        MDPAction.set_description({
            'out': {'value': [-0.004]},
            'in': {'value': [0.004]},
            'kick': {'value': [-1.0]}
        })

        ts = TaskSpec(discount_factor=0.99, reward_range=(-20, 20))
        ts.set_episodic()
        ts.set_charcount_obs(0)
        ts.add_double_act((-1.0, 0.004))

        self._ts_extra += "ACTIONDESCR %s " % str(MDPAction.description)
        self._ts_extra += "COPYRIGHT Penaltykick (Python) implemented by Astrid Jackson"
        ts.set_extra(self._ts_extra)
        return ts.to_taskspec()

    def setup(self):
        if self._client is not None:
            self._client.reset()

    def start(self):
        return Observation()

    def step(self, action):
        return_ro = Reward_observation_terminal()
        return_ro.r = self._calculate_reward(action)
        return_ro.o = Observation()
        return_ro.terminal = self._check_terminal(action)

        return return_ro

    def cleanup(self):
        if self._client is not None:
            self._client.close()

    def message(self, msg):
        if msg == "what is your name?":
            return "my name is skeleton_environment, Python edition!"
        else:
            return "I don't know how to respond to your message"

    # noinspection PyMethodMayBeStatic
    def _check_terminal(self, action):
        if action.doubleArray == [-2.0] or action.doubleArray == MDPAction.description["kick"]["value"]:
            return True
        return False

    def _calculate_reward(self, action):
        reward = -1.0
        if action.doubleArray == MDPAction.description["kick"]["value"]:
            result = self._client.query("check goal")
            if result == "success":
                reward = 20
            elif result == "failure":
                reward = -2
            else:
                raise ValueError("Unknown result from server")
        elif action.doubleArray == [-2.0]:
            reward = -20

        return reward


if __name__ == "__main__":
    loader.load_environment(PenaltyKickEnvironment())
