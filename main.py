# main.py

import argparse
from parameter_gui import create_gui
import json
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_portfolio_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings
import sys
from types import SimpleNamespace
import traceback

warnings.filterwarnings('ignore')

class GUILogger:
    def __init__(self, update_func):
        self.update_func = update_func

    def write(self, message):
        self.update_func(message)

    def flush(self):
        pass

def get_device(requested_device):
    if torch.cuda.is_available() and 'cuda' in requested_device:
        return requested_device
    print(f"CUDA is not available. Using CPU instead of {requested_device}")
    return 'cpu'

import yfinance as yf

def load_price_data(path=None, T=200, N=5, use_yfinance=False, tickers=None):
    """Load or simulate price data."""
    if use_yfinance:
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        df = yf.download(tickers, period="1y")['Adj Close'].dropna()
        prices = df.values
        if prices.shape[0] > T:
            prices = prices[-T:]
        # If less than N tickers returned, pad with last column or truncate
        if prices.shape[1] < N:
            last_col = prices[:, -1].reshape(-1,1)
            prices = np.hstack([prices] + [last_col]*(N - prices.shape[1]))
        elif prices.shape[1] > N:
            prices = prices[:, :N]
        return prices
    if path and os.path.exists(path):
        return np.loadtxt(path, delimiter=',')  # shape: (T, N)
    # Simulate random walk price data if path not provided
    prices = np.cumsum(np.random.randn(T, N) * 2 + 100, axis=0)
    return prices

def main(args, update_output, stop_training):
    sys.stdout = GUILogger(update_output)
    sys.stderr = GUILogger(update_output)

    print("Starting training with parameters:")
    print(json.dumps(args, indent=2))

    args_obj = SimpleNamespace(**args)
    args_obj.device = get_device(args_obj.device)

    # Default tickers
    tickers = getattr(args_obj, 'tickers', ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'])

    # Load prices and pass to environment
    prices = load_price_data(
        path=getattr(args_obj, "price_path", None),
        use_yfinance=getattr(args_obj, "use_yfinance", False),
        tickers=tickers
    )
    env, train_envs, test_envs = make_portfolio_env(
        prices=prices,
        training_num=args_obj.training_num,
        test_num=args_obj.test_num
    )

    args_obj.state_shape = env.observation_space.shape[0]
    try:
        args_obj.action_shape = env.action_space.n  # discrete case
    except AttributeError:
        args_obj.action_shape = env.action_space.shape[0]  # continuous case

    args_obj.max_action = 1.
    args_obj.exploration_noise *= args_obj.max_action

    actor_net = MLP(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape
    )
    actor = Diffusion(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape,
        model=actor_net,
        max_action=args_obj.max_action,
        beta_schedule=args_obj.beta_schedule,
        n_timesteps=args_obj.n_timesteps,
        bc_coef=args_obj.bc_coef
    ).to(args_obj.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args_obj.actor_lr, weight_decay=args_obj.wd)

    critic = DoubleCritic(args_obj.state_shape, args_obj.action_shape).to(args_obj.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args_obj.critic_lr, weight_decay=args_obj.wd)

    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args_obj.logdir, args_obj.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args_obj))
    logger = TensorboardLogger(writer)

    policy = DiffusionOPT(
        args_obj.state_shape,
        actor,
        actor_optim,
        args_obj.action_shape,
        critic,
        critic_optim,
        args_obj.device,
        tau=args_obj.tau,
        gamma=args_obj.gamma,
        estimation_step=args_obj.n_step,
        lr_decay=args_obj.lr_decay,
        lr_maxt=args_obj.epoch,
        bc_coef=args_obj.bc_coef,
        action_space=env.action_space,
        exploration_noise=args_obj.exploration_noise,
    )

    if args_obj.resume_path:
        ckpt = torch.load(args_obj.resume_path, map_location=args_obj.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from:", args_obj.resume_path)

    if args_obj.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args_obj.buffer_size,
            buffer_num=len(train_envs),
            alpha=args_obj.prior_alpha,
            beta=args_obj.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(args_obj.buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def train_callback(epoch: int, env_step: int, **kwargs):
        print(f"Epoch: {epoch}, Env Step: {env_step}")

    def stop_fn(reward, **kwargs):
        if stop_training():
            print(f"Training stopped by user. Best reward: {reward}")
            return True
        return False

    if not args_obj.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args_obj.epoch,
            args_obj.step_per_epoch,
            args_obj.step_per_collect,
            args_obj.test_num,
            args_obj.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
            stop_fn=stop_fn,
            train_fn=train_callback
        )
        pprint.pprint(result)

    print("Training finished.")

    if args_obj.watch:
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

if __name__ == '__main__':
    root, start_training, update_output, stop_training = create_gui()
    root.mainloop()
