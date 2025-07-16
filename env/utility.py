import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def rayleigh_channel_gain(ex, sta):
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    gain = np.abs(gain) ** 2
    return gain

# Function to implement water filling algorithm for power allocation
def water(s, total_power):
    g_n = s
    N_0 = 1
    L = 0
    U = total_power + N_0 * np.sum(1 / (g_n + 1e-6))
    precision = 1e-6

    while U - L > precision:
        alpha_bar = (L + U) / 2
        p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)
        P = np.sum(p_n)
        if P > total_power:
            U = alpha_bar
        else:
            L = alpha_bar

    p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)
    SNR = g_n * p_n_final / N_0
    data_rate = np.log2(1 + SNR)
    expert = p_n_final / total_power
    subexpert = expert + np.random.normal(0, 0.1, len(p_n_final))
    return expert, np.sum(data_rate), subexpert

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution, total_power=10):
    actions = torch.abs(torch.tensor(Aution, dtype=torch.float32))
    normalized_weights = actions / torch.sum(actions)
    a = normalized_weights.numpy() * total_power

    g_n = State
    SNR = g_n * a
    data_rate = np.log2(1 + SNR)

    expert_action, sumdata_rate, subopt_expert_action = water(g_n, total_power)
    reward = np.sum(data_rate) - sumdata_rate

    return reward, expert_action, subopt_expert_action, a / total_power