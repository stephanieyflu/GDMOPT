import numpy as np
import cvxpy as cp
def mean_variance_optimizer(mu, cov, allow_short=False, risk_aversion=1.0):
    n = len(mu)

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    # Ensure PSD by shifting
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = np.min(eigvals)
    if min_eig < 0:
        cov += np.eye(n) * (-min_eig + 1e-6)

    # Tell CVXPY to treat cov as PSD
    cov_psd = cp.psd_wrap(cov)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, cov_psd))
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)

    return w.value if w.value is not None else np.ones(n) / n

def black_litterman_prior(mu, cov, P=None, Q=None, tau=0.025):
    """
        Simple BL using the tau*Sigma shrinkage, no view uncertainty for now.
    """
    if P is None or Q is None:
        return mu  # No views -> revert to prior mean

    omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
    inv_term = np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(omega) @ P
    adjusted_mu = np.linalg.inv(inv_term) @ (
        np.linalg.inv(tau * cov) @ mu + P.T @ np.linalg.inv(omega) @ Q
    )
    return adjusted_mu

def top_k(weights, k=10):
    top_indices = np.argsort(weights)[-k:]
    sparse_weights = np.zeros_like(weights)
    sparse_weights[top_indices] = weights[top_indices]
    return sparse_weights / sparse_weights.sum()

def CompUtility(price_relatives, action, total_weight=1.0, method='sharpe',
                past_returns=None, risk_aversion=0.1, expert_type='equal', cov_estimator='empirical'):
    """
        Compute reward and expert actions for portfolio optimization.
    """

    action = np.clip(action, 0, 1)
    sum_action = np.sum(action)
    weights = action / sum_action if sum_action > 1e-8 else np.ones_like(action) / len(action)
    weights *= total_weight

    price_relatives = np.clip(price_relatives, 1e-6, None)
    portfolio_return = np.dot(weights, price_relatives)

    N = len(price_relatives)

    # Historical returns for expert estimators
    if past_returns is not None and len(past_returns) >= 2:
        X = past_returns[-60:]  # use last 60 days
        mu = np.mean(X - 1, axis=0)  # expected excess returns
        cov = np.cov((X - 1).T)  # sample covariance

        if expert_type == 'mv':
            expert_action = mean_variance_optimizer(mu, cov, allow_short=False, risk_aversion=risk_aversion)
            expert_action = top_k(expert_action, k=10)
        elif expert_type == 'bl':
            # simple view: top assets will outperform
            P = np.eye(N)[np.argsort(mu)[-5:]]  # top 5 views
            Q = mu[np.argsort(mu)[-5:]] + 0.01  # slight bump
            bl_mu = black_litterman_prior(mu, cov, P, Q)
            expert_action = mean_variance_optimizer(bl_mu, cov, allow_short=False, risk_aversion=risk_aversion)
            expert_action = top_k(expert_action, k=10)
        else:
            expert_action = np.ones(N) / N  # default to equal-weight
            expert_action = top_k(expert_action, k=10)
    else:
        expert_action = np.ones(N) / N
        expert_action = top_k(expert_action, k=10)

    sub_expert_action = expert_action + np.random.normal(0, 0.01, N)
    sub_expert_action = np.clip(sub_expert_action, 0, 1)
    sub_expert_action /= np.sum(sub_expert_action)

    if method == 'log':
        reward = np.log1p(portfolio_return - 1)
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
        raise ValueError(f"Unknown reward method: '{method}'.")

    return reward, expert_action, sub_expert_action, weights
