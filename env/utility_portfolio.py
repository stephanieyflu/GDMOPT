import numpy as np

def CompUtility(price_relatives, action, total_weight=1.0, method='sharpe',
                past_returns=None, risk_aversion=0.1):
    """
    Parameters:
        price_relatives (np.array): Current price / previous price for each asset (shape: [N])
        action (np.array): Action vector (portfolio weights)
        total_weight (float): Total portfolio allocation (default 1.0)
        method (str): One of {'log', 'excess', 'sharpe'}
        past_returns (np.array): (T, N) array of past price relatives (for sharpe)
        risk_aversion (float): Penalty weight for risk (used in sharpe)

    Returns:
        reward (float): scalar reward based on selected method
        expert_action (np.array): equal-weight portfolio
        sub_expert_action (np.array): noisy expert action
        real_action (np.array): normalized weights
    """

    # Normalize action
    weights = np.clip(action, 0, 1)
    weights = weights / (np.sum(weights) + 1e-8) * total_weight

    # Portfolio return
    portfolio_return = np.dot(weights, price_relatives)

    # Expert action: equal weight
    N = len(price_relatives)
    expert_action = np.ones(N) / N
    sub_expert_action = expert_action + np.random.normal(0, 0.01, N)
    sub_expert_action = np.clip(sub_expert_action, 0, 1)
    sub_expert_action /= np.sum(sub_expert_action)

    # Reward strategies
    if method == 'log':
        reward = np.log(portfolio_return + 1e-8)

    elif method == 'excess':
        benchmark_return = np.mean(price_relatives)
        reward = np.log(portfolio_return + 1e-8) - np.log(benchmark_return + 1e-8)

    elif method == 'sharpe':
        expected_return = np.log(portfolio_return + 1e-8)
        if past_returns is not None:
            past_portfolio_returns = past_returns @ weights
            risk = np.std(np.log(past_portfolio_returns + 1e-8))
        else:
            risk = 0
        reward = expected_return - risk_aversion * risk

    else:
        raise ValueError(f"Invalid reward method: {method}")

    return reward, expert_action, sub_expert_action, weights
