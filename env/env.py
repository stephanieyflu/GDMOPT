from datetime import datetime
import gym
from gym.spaces import Box
import numpy as np
import pandas as pd
from tianshou.env import DummyVectorEnv
from .utility import CompUtility


class PortfolioEnv(gym.Env):
    def __init__(
        self,
        price_series: np.ndarray,
        reward_method: str = 'sharpe',
        risk_aversion: float = 0.1,
        max_episode_steps: int = None,
        start_idx: int = 0,
        episode_length: int = None,
        log_id: str = None,
        expert_type: str = 'bl',
    ):
        """
        A financial portfolio optimization environment.

        Args:
            price_series (np.ndarray): Shape (T, N_assets). Asset prices over time.
            reward_method (str): Reward calculation method. Options: 'sharpe', 'log', 'excess'.
            risk_aversion (float): Weight on risk penalty.
            max_episode_steps (int): Optional cap on episode length.
            start_idx (int): Starting timestep index in price_series for this episode.
            episode_length (int): Episode length override; if None uses max_episode_steps or remaining data.
        """
        super().__init__()
        self.price_series = price_series
        self.n_assets = price_series.shape[1]
        self.T = price_series.shape[0]

        self.reward_method = reward_method
        self.risk_aversion = risk_aversion
        self.start_idx = start_idx
        self.max_episode_steps = max_episode_steps or (self.T - start_idx - 1)
        self.episode_length = episode_length or self.max_episode_steps

        self.action_space = Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

        self._seed = None

        self.log_id = log_id
        self.expert_type = expert_type

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.episode_logs = []  # For saving info per timestep

        self.current_step = self.start_idx + 1  # current_step is the index in price_series
        self.done = False
        self.portfolio_value = 1.0
        self.history_returns = []

        self.state = self._get_returns(self.current_step)
        return self.state, {"portfolio_value": self.portfolio_value}

    def _get_returns(self, step: int):
        # Since price_series is price relatives, returns are price relatives minus 1
        return self.price_series[step] - 1

    def step(self, action: np.ndarray):
        assert not self.done, "Episode has terminated"

        def apply_cardinality(weights, k=10):
            # Zero all but top-k weights
            top_indices = np.argsort(weights)[-k:]
            sparse = np.zeros_like(weights)
            sparse[top_indices] = weights[top_indices]
            if np.sum(sparse) > 0:
                sparse /= np.sum(sparse)
            else:
                sparse = np.ones_like(weights) / len(weights)
            return sparse

        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.ones_like(action) / self.n_assets  # fallback

        # Apply Top-10 constraint
        action = apply_cardinality(action, k=10)

        prev_prices = self.price_series[self.current_step - 1]
        curr_prices = self.price_series[self.current_step]
        returns = self._get_returns(self.current_step)

        # DEBUG: Print price info
        print(f'\n#------ TIME STEP {self.current_step} ------#')
        print(f'Previous Prices: {prev_prices}')
        print(f'Current Prices:  {curr_prices}')
        print(f'Returns:         {returns}')

        past_returns = None
        if self.reward_method == 'sharpe' and len(self.history_returns) >= 2:
            past_returns = np.array(self.history_returns[-20:])  # trailing 20-day window

        reward, expert_action, sub_expert_action, real_action = CompUtility(
            price_relatives=returns + 1,
            action=action,
            total_weight=1.0,
            method=self.reward_method,
            past_returns=(np.array(self.history_returns) + 1) if past_returns is not None else None,
            risk_aversion=self.risk_aversion,
            expert_type=self.expert_type,
        )

        # Portfolio value update
        portfolio_return = np.dot(real_action, returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.history_returns.append(returns)

        self.episode_logs.append({
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "prev_prices": prev_prices.tolist(),
            "curr_prices": curr_prices.tolist(),
            "returns": returns.tolist(),
            "action": action.tolist(),
            "expert_action": expert_action.tolist(),
            "sub_expert_action": sub_expert_action.tolist(),
            "real_action": real_action.tolist(),
            "reward": reward
        })

        self.current_step += 1

        # Episode termination conditions:
        # 1) Exceeded price_series length
        # 2) Reached episode_length from start_idx
        if (self.current_step >= self.T) or (self.current_step >= self.start_idx + self.episode_length):
            self.done = True

            df = pd.DataFrame(self.episode_logs)
            log_filename = f"episode_log_{self.log_id or self.start_idx}.csv"
            log_path = rf"C:\Users\inula\OneDrive\ドキュメント\GitHub\GDMOPT\log\data\{log_filename}"
            df.to_csv(log_path, index=False)

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


def make_portfolio_env(prices: np.ndarray,
                       training_num: int = 0,
                       test_num: int = 0,
                       reward_method: str = 'sharpe',
                       episode_length: int = 100):
    """
    Creates portfolio environments using price relatives (ratios) instead of normalized returns.

    Args:
        prices: np.ndarray of shape (T, N_assets) with raw asset prices (non-negative).
        training_num: Number of parallel training envs.
        test_num: Number of parallel test envs.
        reward_method: Reward calculation method.
        episode_length: Length of each episode.
    """

    # Compute price relatives: ratio of price at t to price at t-1
    price_relatives = prices[1:] / np.clip(prices[:-1], 1e-8, np.inf)  # shape (T-1, N_assets)
    # price_relatives are always >= 0 (assuming prices >= 0)

    def get_env(start_idx=None):
        if start_idx is None:
            max_start = price_relatives.shape[0] - episode_length - 1
            start_idx_ = np.random.randint(0, max_start) if max_start > 0 else 0
        else:
            start_idx_ = start_idx
        return PortfolioEnv(
            price_relatives,
            reward_method=reward_method,
            max_episode_steps=episode_length,
            start_idx=start_idx_,
            episode_length=episode_length,
            log_id=f"epoch_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    env = get_env(start_idx=0)
    train_envs = DummyVectorEnv([lambda s=i: get_env() for i in range(training_num)]) if training_num else None
    test_envs = DummyVectorEnv([lambda s=i: get_env() for i in range(test_num)]) if test_num else None

    return env, train_envs, test_envs
