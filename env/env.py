import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np

class AIGCEnv(gym.Env):

    def __init__(self):
        self._flag = 0
        self._num_channels = 3
        self._max_power = 10

        # Observation: 3 channel gains + 1 reward placeholder
        self._observation_space = Box(low=0, high=np.inf, shape=(self._num_channels + 1,), dtype=np.float32)

        # Action: continuous power allocation vector over 3 channels (can be normalized)
        self._action_space = Box(low=0, high=1, shape=(self._num_channels,), dtype=np.float32)

        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self._steps_per_episode = 9 # number of denoising steps

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        # Channel gains: 3 values in [0.5, 2.5]
        self.channel_gains = np.random.uniform(0.5, 2.5, self._num_channels)

        # Append reward placeholder (0)
        reward_placeholder = [0]
        state = np.concatenate([self.channel_gains, reward_placeholder])

        self._laststate = state
        return state

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        reward, expert_action, sub_expert_action, real_action = CompUtility(
            self.channel_gains, action, total_power=self._max_power
        )

        self._laststate[-1] = reward
        self._laststate[0:-1] = self.channel_gains * real_action

        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        info = {
            'num_steps': self._num_steps,
            'expert_action': expert_action,
            'sub_expert_action': sub_expert_action
        }

        return self._laststate, reward, self._terminated, info

    def reset(self):
        self._num_steps = 0
        self._terminated = False
        return self.state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)

    return env, train_envs, test_envs
