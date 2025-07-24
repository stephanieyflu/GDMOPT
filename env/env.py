import gym
from gym.spaces import Box
import numpy as np
from tianshou.env import DummyVectorEnv

from .utility import CompUtility  # Assumes this is a reward-calculating function


class PortfolioEnv(gym.Env):
    def __init__(
        self,
        price_series: np.ndarray,
        reward_method: str = 'sharpe',
        risk_aversion: float = 0.1,
        max_episode_steps: int = None,
    ):
        """
        A financial portfolio optimization environment.

        Args:
            price_series (np.ndarray): Shape (T, N_assets). Asset prices over time.
            reward_method (str): Reward calculation method. Options: 'sharpe', 'log', 'excess'.
            risk_aversion (float): Weight on risk penalty.
            max_episode_steps (int): Optional cap on episode length.
        """
        super().__init__()
        self.price_series = price_series
        self.n_assets = price_series.shape[1]
        self.T = price_series.shape[0]

        self.reward_method = reward_method
        self.risk_aversion = risk_aversion
        self.max_episode_steps = max_episode_steps or self.T - 1

        self.action_space = Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

        self._seed = None
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.current_step = 1
        self.done = False
        self.portfolio_value = 1.0
        self.history_returns = []

        self.state = self._get_returns(self.current_step)
        return self.state, {"portfolio_value": self.portfolio_value}

    def _get_returns(self, step: int):
        prev_prices = self.price_series[step - 1]
        curr_prices = self.price_series[step]
        returns = (curr_prices - prev_prices) / np.clip(prev_prices, 1e-8, np.inf)
        return returns

    def step(self, action: np.ndarray):
        assert not self.done, "Episode has terminated"

        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.ones_like(action) / self.n_assets  # fallback to uniform

        returns = self._get_returns(self.current_step)

        past_returns = None
        if self.reward_method == 'sharpe' and len(self.history_returns) >= 2:
            past_returns = np.array(self.history_returns[-20:])  # trailing 20-day window

        reward, expert_action, sub_expert_action, real_action = CompUtility(
            price_relatives=returns + 1,
            action=action,
            total_weight=1.0,
            method=self.reward_method,
            past_returns=(np.array(self.history_returns) + 1) if past_returns is not None else None,
            risk_aversion=self.risk_aversion
        )

        # Portfolio value update
        portfolio_return = np.dot(real_action, returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.history_returns.append(returns)

        self.current_step += 1
        if self.current_step >= self.T or self.current_step >= self.max_episode_steps:
            self.done = True
            next_state = np.zeros(self.n_assets)
        else:
            next_state = self._get_returns(self.current_step)

        self.state = next_state

        info = {
            "portfolio_value": self.portfolio_value,
            "expert_action": expert_action,
            "sub_expert_action": sub_expert_action,
            "real_action": real_action
        }

        print(f'#-------- INFO --------# {info}')
        return self.state, reward, self.done, info

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)


def make_portfolio_env(prices: np.ndarray, training_num: int = 0, test_num: int = 0, reward_method: str = 'sharpe'):
    """Creates single, training, and test environments."""
    def get_env():
        return PortfolioEnv(prices, reward_method=reward_method)

    env = get_env()
    train_envs = DummyVectorEnv([get_env for _ in range(training_num)]) if training_num else None
    test_envs = DummyVectorEnv([get_env for _ in range(test_num)]) if test_num else None

    return env, train_envs, test_envs
