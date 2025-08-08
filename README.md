# 📈 Portfolio Optimization with Generative Diffusion Models

This repository contains the code from my Summer 2025 research project at the University of Osaka’s Inuiguchi Lab, as part of the FrontierLab Mini Program. The work explores the application of **generative diffusion models (GDMs)** to **portfolio optimization** - a high-dimensional, constrained, and dynamic decision problem central to financial engineering.

> ✅ Inspired by techniques originally developed for wireless communication networks (Du et al., 2024), we adapt a generative modeling + deep reinforcement learning (DRL) framework to financial markets.

## 🧠 Core Idea

Traditional portfolio optimization methods like mean-variance optimization often:
- Rely on strong assumptions (Gaussian returns, convexity)
- Struggle under non-stationary or noisy market conditions
- Fail to incorporate real-world constraints (e.g. no shorting, cardinality)

**Generative Diffusion Models** offer:
- A data-driven way to generate portfolio allocations from return/covariance estimates
- Implicit learning of constraints and market structure
- Strong generalization in non-convex, multi-modal optimization problems

## 🧪 Project Highlights

### 📊 Simulated Financial Environment
- State space: simulated or historical expected returns (`mu`) and standard deviations (`sigma`)
- Dynamic environments via rolling historical windows
- Return/covariance matrices updated at each time step

### 🎯 Action & Reward Design
- Actions = portfolio weights, normalized and optionally pruned by cardinality
- Reward = difference between learned Sharpe Ratio and expert benchmark
- Supports alternate reward functions like log-utility

### 🧠 Learning Architecture
- Generative Diffusion Model (GDM) for sampling portfolio weights
- Evaluation network for computing utility of generated actions
- Expert comparison using classical mean-variance theory

## 📁 Repository Structure

```bash
.
├── diffusion/          # Generative model implementation
├── data/               # (Optional) Historical asset price datasets
├── env/                # Financial simulation environment
├── log/                # Code logs for training and evaluation runs
├── policy/             # 
├── notebooks/          # Visualizations and exploratory analysis
├── Software/           # 
├── static/             # 
├── main.py             # Entrypoint for training/inference
├── parameter_gui.py    # 
├── requirements.txt
└── README.md
````

## 📈 Results

Key experiments included:

* Comparing learned vs. 'expert' Sharpe ratios
* Analyzing action quality under different levels of noise/volatility
* Performance under cardinality and sector constraints

## 🏁 Getting Started

See original GitHub repository [here](https://github.com/HongyangDu/GDMOPT).

    pip install pandas cvxpy

## 📚 Reference

* Du, Y., Zhou, D., Liu, X., & Zhang, R. (2024). *Generative Diffusion Models Meet Deep Reinforcement Learning for Network Optimization*. [arXiv](https://arxiv.org/abs/2402.00000)

## 📌 Acknowledgements

Thanks to Professor **Naoki Hayashi** and the **Inuiguchi Lab** for their guidance and mentorship. This project was conducted as part of the **Frontier MiniLab Program (Summer 2025)**.


default\diffusion\Aug08-020523
expert_type = equal
reward_method = sharpe
prices = generate_gbm_prices(S0=100, mu=0.05, sigma=0.2, T=1000, N=30)

default\diffusion\Aug08-020135
expert_type = mv
reward_method = sharpe
prices = generate_gbm_prices(S0=100, mu=0.05, sigma=0.2, T=1000, N=30)

default\diffusion\Aug08-080613
expert_type = bl
reward_method = sharpe
prices = generate_gbm_prices(S0=100, mu=0.05, sigma=0.2, T=1000, N=30)

default\diffusion\Aug08-094433
expert_type = equal
reward_method = sharpe
prices.csv

default\diffusion\Aug08-094731
expert_type = mv
reward_method = sharpe
prices.csv

default\diffusion\Aug08-094820
expert_type = bl
reward_method = sharpe
prices.csv
