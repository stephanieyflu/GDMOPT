import numpy as np

def CompUtility(price_relatives, action, total_weight=1.0, method='sharpe',
                past_returns=None, risk_aversion=0.1):
    """
    Compute reward and expert actions for portfolio optimization.

    Args:
        price_relatives (np.ndarray): Array of shape (N,) representing price_relatives (i.e., current_price / prev_price).
        action (np.ndarray): Array of shape (N,) representing the agent's raw action (unconstrained weights).
        total_weight (float): Total allocation weight. Defaults to 1.0.
        method (str): Reward strategy - one of {'log', 'excess', 'sharpe'}.
        past_returns (np.ndarray): Optional (T, N) array of historical price relatives.
        risk_aversion (float): Penalization factor for volatility in 'sharpe' reward.

    Returns:
        reward (float): Scalar reward for the given action.
        expert_action (np.ndarray): Equal-weight baseline portfolio.
        sub_expert_action (np.ndarray): Noisy version of expert action.
        real_action (np.ndarray): Final normalized portfolio weights derived from action input.
    """

    # Step 1: Normalize the agent's action into valid portfolio weights
    action = np.clip(action, 0, 1)
    sum_action = np.sum(action)
    if sum_action < 1e-8:
        weights = np.ones_like(action) / len(action)  # fallback to equal weight
    else:
        weights = action / sum_action
    weights *= total_weight

    # Step 2: Compute the portfolio return
    price_relatives = np.clip(price_relatives, 1e-6, None)  # avoid division by zero or log(0)
    portfolio_return = np.dot(weights, price_relatives)

    # Step 3: Define expert and sub-expert actions
    N = len(price_relatives)
    expert_action = np.ones(N) / N
    sub_expert_action = expert_action + np.random.normal(0, 0.01, N)
    sub_expert_action = np.clip(sub_expert_action, 0, 1)
    sub_expert_action /= np.sum(sub_expert_action)

    # Step 4: Compute reward based on specified method
    if method == 'log':
        reward = np.log1p(portfolio_return - 1)  # log(portfolio_return)

    elif method == 'excess':
        benchmark_return = np.mean(price_relatives)
        reward = np.log1p(portfolio_return - 1) - np.log1p(benchmark_return - 1)

    elif method == 'sharpe':
        expected_log_return = np.log1p(portfolio_return - 1)

        if past_returns is not None and len(past_returns) > 1:
            past_returns = np.clip(past_returns, 1e-6, None)
            past_portfolio_returns = past_returns @ weights
            log_returns = np.log1p(past_portfolio_returns - 1)
            volatility = np.std(log_returns)
        else:
            volatility = 0.0

        reward = expected_log_return - risk_aversion * volatility

    else:
        raise ValueError(f"Unknown reward method: '{method}'. Choose from ['log', 'excess', 'sharpe'].")

    return reward, expert_action, sub_expert_action, weights
