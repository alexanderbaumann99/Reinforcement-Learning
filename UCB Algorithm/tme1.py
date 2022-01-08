"""
n-bandits problem
Reward immédiat
Etat courant ne dépend pas des actions passées
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv

import seaborn as sns
import matplotlib.pyplot as plt


# <numero de l'article>:<représentation de l'article en 5 dimensions séparées par des ";">:<taux de
# clics sur les publicités de 10 annonceurs séparés par des ";">
data = pd.read_csv("CTR.txt", sep=":|;", header=None, index_col=0)
n, c = data.shape


def random_strategy():
    """Choose advertiser randomly"""
    select = np.random.randint(5, 15, n)
    reward = np.zeros(n)
    for i, s in enumerate(select):
        reward[i] = data.iloc[i, s]
    total_reward_random = np.cumsum(reward)
    return total_reward_random


def static_best_strategy():
    """Always choose the advertiser with the best accumulated click-through rate (cheat!)"""
    scores = data.iloc[:, 5:15]
    scores_cumulative = scores.sum()
    best_ann = scores_cumulative.idxmax()
    reward = scores.iloc[:, best_ann]
    total_reward_best = np.cumsum(reward).to_numpy()
    return total_reward_best


def optimal_strategy():
    """Choose the best advertiser with the best click-through rate"""
    scores = data.iloc[:, 5:15]
    optimal_score = scores.max(axis=1)
    total_reward_optimal = np.cumsum(optimal_score)
    return total_reward_optimal


# In the reality, when we do the decision for the time n, we can only se the click-through rate of [1...n-1] decisions.

def ucb():
    """Upper confidence bound"""
    scores = data.iloc[:, 5:15]
    # upper-confidence Bound table
    B = np.zeros((n, 10))
    # number of selected time of annoncer
    count = np.zeros(10)
    # total reward of annoncer
    total_reward_empirique = np.zeros(10)
    ucb_reward = np.zeros(n)
    # we initialise first 10 advertiser decision with advertiser number e.g. 1->1, 2->1,..., 10->10
    for i in range(10):
        B[i, i] = 1
        count[i] += 1
        total_reward_empirique[i] += scores.iloc[i, i]
        ucb_reward[i] = scores.iloc[i, i]
    # devision after the initialisation
    for i in np.arange(10, n):
        B[i] = total_reward_empirique / count + np.sqrt((2 * np.log(i + 1)) / count)
        c = B[i].argmax()
        s = scores.iloc[i, c]
        ucb_reward[i] = s
        total_reward_empirique[c] += s
        count[c] += 1
    total_ucb_reward = np.cumsum(ucb_reward)
    return total_ucb_reward


def lin_ubc(delta=0.1, m=5):
    """This algorithm take the first 5 elements into consideration,
    The first 5 elements are the context and it is useful to predict the variation in observed click-through rate"""
    scores = data.iloc[:, 5:15]
    context = data.iloc[:, 0:5]

    alpha = 1 + np.sqrt(np.log(2/delta)/2)
    A = np.eye(m)
    A = A[None, ...]
    A = np.repeat(A, m, axis=0)
    b = np.zeros((m, m, 1))
    p = np.zeros((n, m))
    a = np.zeros(n, dtype=int)
    lin_ucb_reward = np.zeros(n)
    for i in range(n):
        A_inv = inv(A)
        theta = A_inv @ b
        x_t = context.iloc[i].to_numpy()[..., None]
        p[i] = (np.transpose(theta, (0, 2, 1)) @ x_t + alpha * np.sqrt(x_t.T @ A_inv @ x_t))[:, 0, 0]
        a[i] = np.argmax(p[i])
        lin_ucb_reward[i] = scores.iloc[i, a[i]]
        A[a[i]] = A[a[i]] + x_t @ x_t.T
        b[a[i]] = b[a[i]] + lin_ucb_reward[i] * x_t
    total_lin_ucb_reward = np.cumsum(lin_ucb_reward)
    return total_lin_ucb_reward


# plot the results
if __name__ == '__main__':
    t = np.arange(1, n + 1)
    sns.set()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(t, random_strategy(), label="Total Random reward")
    ax.plot(t, static_best_strategy(), label="Total StaticBest reward")
    ax.plot(t, optimal_strategy(), label="Total Optimal reward")
    ax.plot(t, ucb(), label="Total UCB reward")
    ax.plot(t, lin_ubc(), label="Total Lin UCB reward")
    ax.set_xlabel('articles')
    ax.set_ylabel('reward')
    ax.set_title('n-Bandits application in adviser selection')
    ax.legend()
    plt.show()












