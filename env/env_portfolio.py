import gym
from gym.spaces import Box
import numpy as np
from tianshou.env import DummyVectorEnv

from comp_utility import CompUtility  # or import directly if in same file

class PortfolioEnv(gym.Env):
    def __init__(self, price_series, reward_method='sharpe', risk_aversion=0.1):
        """
        price_series: np.array of shape (T, N_assets), e.g. daily prices
        reward_method: 'log', 'excess', or 'sharpe'
        """
        super().__init__()
        self.price_series = price_series
        self.n_assets = price_series.shape[1]
        self.current_step = 1
        self.done = False
        self.portfolio_value = 1.0
        self.reward_method = reward_method
        self.risk_aversion = risk_aversion

        self.action_space = Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

        self.state = None
        self.history_returns = []

    def reset(self):
        self.current_step = 1
        self.done = False
        self.portfolio_value = 1.0
        self.history_returns = []

        self.state = self._get_returns(self.current_step)
        return self.state

    def _get_returns(self, step):
        prev_prices = self.price_series[step - 1]
        curr_prices = self.price_series[step]
        returns = (curr_prices - prev_prices) / prev_prices
        return returns

    def step(self, action):
        assert not self.done, "Episode has terminated"

        returns = self._get_returns(self.current_step)

        # For sharpe ratio: collect past returns matrix
        if self.reward_method == 'sharpe' and len(self.history_returns) >= 2:
            past_returns = np.array(self.history_returns[-20:])  # last 20 days, e.g.
        else:
            past_returns = None

        reward, expert_action, sub_expert_action, real_action = CompUtility(
            price_relatives=(returns + 1),  # convert simple returns to price relatives
            action=action,
            total_weight=1.0,
            method=self.reward_method,
            past_returns=(np.array(self.history_returns) + 1) if past_returns is not None else None,
            risk_aversion=self.risk_aversion
        )

        self.portfolio_value *= (1 + np.dot(real_action, returns))
        self.history_returns.append(returns)

        self.current_step += 1
        if self.current_step >= len(self.price_series):
            self.done = True
            obs = np.zeros(self.n_assets)
        else:
            obs = self._get_returns(self.current_step)

        return obs, reward, self.done, {
            "portfolio_value": self.portfolio_value,
            "expert_action": expert_action,
            "sub_expert_action": sub_expert_action,
            "real_action": real_action
        }


def make_portfolio_env(prices, training_num=0, test_num=0, reward_method='sharpe'):
    def get_env():
        return PortfolioEnv(prices, reward_method=reward_method)

    env = get_env()
    train_envs = DummyVectorEnv([get_env for _ in range(training_num)]) if training_num else None
    test_envs = DummyVectorEnv([get_env for _ in range(test_num)]) if test_num else None

    return env, train_envs, test_envs

